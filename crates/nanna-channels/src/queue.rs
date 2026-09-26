//! Message queue with rate limiting and retry logic
//!
//! Provides buffered message delivery with:
//! - Per-channel rate limiting with token bucket
//! - Exponential backoff on failures
//! - Persistent queue for offline resilience
//! - Priority queuing (urgent messages first)

use crate::{Channel, ChannelError, OutgoingMessage};
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, warn};

/// Message priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[derive(Default)]
pub enum MessagePriority {
    /// Background messages (e.g., scheduled notifications)
    Low = 0,
    /// Normal conversation messages
    #[default]
    Normal = 1,
    /// Time-sensitive messages (e.g., alerts)
    High = 2,
    /// Critical messages (e.g., error notifications)
    Urgent = 3,
}


/// Queued message with metadata
#[derive(Debug, Clone)]
pub struct QueuedMessage {
    /// The message to send
    pub message: OutgoingMessage,
    /// Priority level
    pub priority: MessagePriority,
    /// When the message was queued
    pub queued_at: Instant,
    /// Number of retry attempts
    pub attempts: u32,
    /// Next retry time (if retrying)
    pub retry_after: Option<Instant>,
    /// Maximum attempts before dropping
    pub max_attempts: u32,
    /// Unique ID for tracking
    pub id: u64,
}

impl PartialEq for QueuedMessage {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for QueuedMessage {}

impl PartialOrd for QueuedMessage {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueuedMessage {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority first, then older messages first
        match self.priority.cmp(&other.priority) {
            std::cmp::Ordering::Equal => other.queued_at.cmp(&self.queued_at),
            ord => ord,
        }
    }
}

/// Rate limiter using token bucket algorithm
#[derive(Debug)]
pub struct RateLimiter {
    /// Tokens available
    tokens: f64,
    /// Maximum tokens (bucket size)
    max_tokens: f64,
    /// Tokens added per second
    refill_rate: f64,
    /// Last refill time
    last_refill: Instant,
    /// Cooldown until (for rate limit errors)
    cooldown_until: Option<Instant>,
}

impl RateLimiter {
    /// Create a new rate limiter
    #[must_use]
    pub fn new(max_tokens: f64, refill_rate: f64) -> Self {
        Self {
            tokens: max_tokens,
            max_tokens,
            refill_rate,
            last_refill: Instant::now(),
            cooldown_until: None,
        }
    }

    /// Create with provider-specific defaults
    #[must_use]
    pub fn for_provider(provider: &str) -> Self {
        match provider {
            "telegram" => Self::new(30.0, 1.0),      // 30 msg/min
            "discord" => Self::new(5.0, 0.2),        // 5 msg burst, 1/5s refill
            "slack" => Self::new(1.0, 1.0),          // 1 msg/s
            "signal" => Self::new(10.0, 0.5),        // 10 burst, 1/2s refill
            "whatsapp" => Self::new(20.0, 0.33),     // 20 msg/min
            _ => Self::new(10.0, 1.0),               // Conservative default
        }
    }

    /// Refill tokens based on elapsed time
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = elapsed.mul_add(self.refill_rate, self.tokens).min(self.max_tokens);
        self.last_refill = now;
    }

    /// Check if we can send (without consuming)
    pub fn can_send(&mut self) -> bool {
        // Check cooldown first
        if let Some(until) = self.cooldown_until {
            if Instant::now() < until {
                return false;
            }
            self.cooldown_until = None;
        }

        self.refill();
        self.tokens >= 1.0
    }

    /// Try to consume a token (returns true if allowed)
    pub fn try_acquire(&mut self) -> bool {
        if !self.can_send() {
            return false;
        }

        self.tokens -= 1.0;
        true
    }

    /// Time until next token is available
    pub fn time_until_available(&mut self) -> Duration {
        // Check cooldown first
        if let Some(until) = self.cooldown_until {
            let now = Instant::now();
            if now < until {
                return until.duration_since(now);
            }
            self.cooldown_until = None;
        }

        self.refill();
        if self.tokens >= 1.0 {
            Duration::ZERO
        } else {
            let needed = 1.0 - self.tokens;
            Duration::from_secs_f64(needed / self.refill_rate)
        }
    }

    /// Set a cooldown period (e.g., from 429 response)
    pub fn set_cooldown(&mut self, duration: Duration) {
        self.cooldown_until = Some(Instant::now() + duration);
        self.tokens = 0.0; // Drain tokens
    }
}

/// Result of a send attempt
#[derive(Debug)]
pub enum SendResult {
    /// Message sent successfully
    Success(String),
    /// Rate limited, retry after duration
    RateLimited(Duration),
    /// Temporary failure, will retry
    RetryLater(Duration, String),
    /// Permanent failure, message dropped
    Failed(String),
}

/// Message queue configuration
#[derive(Debug, Clone)]
pub struct QueueConfig {
    /// Maximum messages in queue per channel
    pub max_queue_size: usize,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Initial retry delay
    pub initial_retry_delay: Duration,
    /// Maximum retry delay
    pub max_retry_delay: Duration,
    /// How long to keep failed messages for inspection
    pub failed_message_ttl: Duration,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 1000,
            max_retries: 5,
            initial_retry_delay: Duration::from_secs(1),
            max_retry_delay: Duration::from_secs(300), // 5 minutes
            failed_message_ttl: Duration::from_secs(3600), // 1 hour
        }
    }
}

/// Queue statistics
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct QueueStats {
    /// Messages currently queued
    pub queued: usize,
    /// Messages sent successfully
    pub sent: u64,
    /// Messages that failed permanently
    pub failed: u64,
    /// Messages currently being retried
    pub retrying: usize,
    /// Current rate limit cooldown (if any)
    pub cooldown_remaining_ms: Option<u64>,
}

/// Per-channel queue state
struct ChannelQueue {
    /// Priority queue of messages
    messages: BinaryHeap<QueuedMessage>,
    /// Rate limiter
    rate_limiter: RateLimiter,
    /// Statistics
    stats: QueueStats,
    /// Next message ID
    next_id: u64,
}

impl ChannelQueue {
    fn new(provider: &str) -> Self {
        Self {
            messages: BinaryHeap::new(),
            rate_limiter: RateLimiter::for_provider(provider),
            stats: QueueStats::default(),
            next_id: 0,
        }
    }
}

/// Message queue manager
pub struct MessageQueue {
    /// Per-channel queues
    queues: Arc<RwLock<HashMap<String, ChannelQueue>>>,
    /// Configuration
    config: QueueConfig,
    /// Channel for queue events
    event_tx: mpsc::Sender<QueueEvent>,
    /// Event receiver (for consumers)
    event_rx: Arc<RwLock<mpsc::Receiver<QueueEvent>>>,
}

/// Events emitted by the queue
#[derive(Debug, Clone)]
pub enum QueueEvent {
    /// Message was queued
    Queued { provider: String, message_id: u64 },
    /// Message was sent
    Sent { provider: String, message_id: u64, result_id: String },
    /// Message send failed (will retry)
    RetryScheduled { provider: String, message_id: u64, attempt: u32, retry_in: Duration },
    /// Message failed permanently
    Failed { provider: String, message_id: u64, error: String },
    /// Rate limited
    RateLimited { provider: String, cooldown: Duration },
    /// Queue stats updated
    StatsUpdated { provider: String, stats: QueueStats },
}

impl MessageQueue {
    /// Create a new message queue
    #[must_use]
    pub fn new(config: QueueConfig) -> Self {
        let (event_tx, event_rx) = mpsc::channel(1000);
        Self {
            queues: Arc::new(RwLock::new(HashMap::new())),
            config,
            event_tx,
            event_rx: Arc::new(RwLock::new(event_rx)),
        }
    }

    /// Publish a queue event without ever waiting.
    ///
    /// Events are advisory, and the channel is bounded (1000) with nobody
    /// obliged to read it. Emitting used to `.await` a `send` while holding
    /// the queues' write lock, so once 1000 events went unread the next
    /// enqueue blocked forever — and with it every other queue operation. A
    /// full channel now drops the event instead (and says so at debug level).
    fn emit(&self, event: QueueEvent) {
        if let Err(mpsc::error::TrySendError::Full(dropped)) = self.event_tx.try_send(event) {
            debug!("Queue event channel full; dropped {:?}", dropped);
        }
    }

    /// Get the event receiver
    #[must_use]
    pub fn events(&self) -> Arc<RwLock<mpsc::Receiver<QueueEvent>>> {
        self.event_rx.clone()
    }

    /// Enqueue a message with default priority
    pub async fn enqueue(&self, message: OutgoingMessage) -> u64 {
        self.enqueue_with_priority(message, MessagePriority::Normal).await
    }

    /// Enqueue a message with specific priority
    pub async fn enqueue_with_priority(
        &self,
        message: OutgoingMessage,
        priority: MessagePriority,
    ) -> u64 {
        let provider = message.channel.provider.clone();
        let mut queues = self.queues.write().await;

        let queue = queues
            .entry(provider.clone())
            .or_insert_with(|| ChannelQueue::new(&provider));

        let id = queue.next_id;
        queue.next_id += 1;

        let entry = QueuedMessage {
            message,
            priority,
            queued_at: Instant::now(),
            attempts: 0,
            retry_after: None,
            max_attempts: self.config.max_retries,
            id,
        };

        // Check queue size limit
        if queue.messages.len() >= self.config.max_queue_size {
            // Drop lowest priority message
            if let Some(dropped) = queue.messages.iter().min().cloned() {
                queue.messages.retain(|m| m.id != dropped.id);
                warn!("Queue full for {}, dropped message {}", provider, dropped.id);
            }
        }

        queue.messages.push(entry);
        queue.stats.queued = queue.messages.len();

        self.emit(QueueEvent::Queued { 
            provider: provider.clone(), 
            message_id: id,
        });

        debug!("Enqueued message {} for {} (priority: {:?})", id, provider, priority);
        // Held across the (synchronous) emit: concurrent enqueues emit their
        // `Queued` events in the same order they assigned message ids.
        drop(queues);
        id
    }

    /// Process the queue for a channel, sending messages that are ready
    pub async fn process<C: Channel + ?Sized>(
        &self,
        provider: &str,
        channel: &C,
    ) -> Vec<SendResult> {
        let mut results = Vec::new();
        let mut to_retry = Vec::new();

        loop {
            // Get next message to send
            let Some(mut msg) = self.take_ready_message(provider).await else {
                break;
            };

            // Acquire rate limit token
            {
                let mut queues = self.queues.write().await;
                if let Some(queue) = queues.get_mut(provider)
                    && !queue.rate_limiter.try_acquire() {
                        // Put message back
                        queue.messages.push(msg);
                        break;
                    }
            }

            // Send the message
            msg.attempts += 1;
            let send_result = channel.send(msg.message.clone()).await;

            match send_result {
                Ok(result_id) => {
                    results.push(SendResult::Success(result_id.clone()));
                    
                    let mut queues = self.queues.write().await;
                    if let Some(queue) = queues.get_mut(provider) {
                        queue.stats.sent += 1;
                        queue.stats.queued = queue.messages.len();
                    }

                    self.emit(QueueEvent::Sent {
                        provider: provider.to_string(),
                        message_id: msg.id,
                        result_id,
                    });
                    // Held across the (synchronous) emit so the `Sent` event is
                    // ordered with the stats change it reports.
                    drop(queues);
                }
                Err(ChannelError::RateLimited) => {
                    // Apply cooldown and requeue
                    let cooldown = self.calculate_backoff(msg.attempts);
                    
                    {
                        let mut queues = self.queues.write().await;
                        if let Some(queue) = queues.get_mut(provider) {
                            queue.rate_limiter.set_cooldown(cooldown);
                        }
                    }

                    msg.retry_after = Some(Instant::now() + cooldown);
                    to_retry.push(msg.clone());

                    results.push(SendResult::RateLimited(cooldown));

                    self.emit(QueueEvent::RateLimited {
                        provider: provider.to_string(),
                        cooldown,
                    });

                    warn!("Rate limited on {}, cooling down for {:?}", provider, cooldown);
                    break;
                }
                Err(e) => {
                    if msg.attempts < msg.max_attempts {
                        // Retry with backoff
                        let delay = self.calculate_backoff(msg.attempts);
                        msg.retry_after = Some(Instant::now() + delay);
                        to_retry.push(msg.clone());

                        results.push(SendResult::RetryLater(delay, e.to_string()));

                        self.emit(QueueEvent::RetryScheduled {
                            provider: provider.to_string(),
                            message_id: msg.id,
                            attempt: msg.attempts,
                            retry_in: delay,
                        });

                        debug!("Will retry message {} in {:?} (attempt {})", msg.id, delay, msg.attempts);
                    } else {
                        // Max retries exceeded
                        results.push(SendResult::Failed(e.to_string()));

                        let mut queues = self.queues.write().await;
                        if let Some(queue) = queues.get_mut(provider) {
                            queue.stats.failed += 1;
                            queue.stats.queued = queue.messages.len();
                        }

                        self.emit(QueueEvent::Failed {
                            provider: provider.to_string(),
                            message_id: msg.id,
                            error: e.to_string(),
                        });

                        error!("Message {} failed after {} attempts: {}", msg.id, msg.attempts, e);
                        // Held across the (synchronous) emit so the `Failed` event
                        // is ordered with the stats change it reports.
                        drop(queues);
                    }
                }
            }
        }

        // Re-add messages that need retry
        if !to_retry.is_empty() {
            let mut queues = self.queues.write().await;
            if let Some(queue) = queues.get_mut(provider) {
                for msg in to_retry {
                    queue.messages.push(msg);
                }
                queue.stats.retrying = queue.messages.iter().filter(|m| m.attempts > 0).count();
                queue.stats.queued = queue.messages.len();
            }
        }

        results
    }

    /// Pop the highest-priority message whose retry time has come, or `None`
    /// when the provider has no queue, is rate limited, or has nothing ready.
    async fn take_ready_message(&self, provider: &str) -> Option<QueuedMessage> {
        let mut queues = self.queues.write().await;
        let queue = queues.get_mut(provider)?;

        // Check rate limit
        if !queue.rate_limiter.can_send() {
            let wait = queue.rate_limiter.time_until_available();
            if wait > Duration::ZERO {
                // A cooldown beyond u64::MAX milliseconds (~584 million years)
                // cannot be scheduled, so saturating never differs in practice.
                queue.stats.cooldown_remaining_ms =
                    Some(u64::try_from(wait.as_millis()).unwrap_or(u64::MAX));
                return None;
            }
        }
        queue.stats.cooldown_remaining_ms = None;

        // Get highest priority message that's ready
        let now = Instant::now();
        let ready_idx = queue.messages.iter().position(|m| {
            m.retry_after.is_none_or(|t| now >= t)
        });

        let found = if ready_idx.is_some() {
            // Pop the top message (highest priority that's ready)
            let mut temp = Vec::new();
            let mut found = None;
            while let Some(m) = queue.messages.pop() {
                if found.is_none() && m.retry_after.is_none_or(|t| now >= t) {
                    found = Some(m);
                } else {
                    temp.push(m);
                }
            }
            for m in temp {
                queue.messages.push(m);
            }
            found
        } else {
            None
        };
        drop(queues);
        found
    }

    /// Calculate exponential backoff delay
    fn calculate_backoff(&self, attempts: u32) -> Duration {
        let base = self.config.initial_retry_delay.as_secs_f64();
        let max = self.config.max_retry_delay.as_secs_f64();
        
        // Exponential backoff with jitter. `attempts` counts real send attempts,
        // so it never approaches `i32::MAX`; saturating is unreachable.
        let exponent = i32::try_from(attempts).unwrap_or(i32::MAX) - 1;
        let delay = (base * 2.0_f64.powi(exponent)).min(max);
        let jitter = delay * 0.1 * rand::random::<f64>();
        
        Duration::from_secs_f64(delay + jitter)
    }

    /// Get queue statistics for a provider
    pub async fn stats(&self, provider: &str) -> Option<QueueStats> {
        let queues = self.queues.read().await;
        queues.get(provider).map(|q| q.stats.clone())
    }

    /// Get statistics for all providers
    pub async fn all_stats(&self) -> HashMap<String, QueueStats> {
        let queues = self.queues.read().await;
        queues.iter().map(|(k, v)| (k.clone(), v.stats.clone())).collect()
    }

    /// Clear the queue for a provider
    pub async fn clear(&self, provider: &str) {
        let mut queues = self.queues.write().await;
        if let Some(queue) = queues.get_mut(provider) {
            queue.messages.clear();
            queue.stats.queued = 0;
            queue.stats.retrying = 0;
        }
    }

    /// Get queue length for a provider
    pub async fn len(&self, provider: &str) -> usize {
        let queues = self.queues.read().await;
        queues.get(provider).map_or(0, |q| q.messages.len())
    }
}

impl Default for MessageQueue {
    fn default() -> Self {
        Self::new(QueueConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChannelId;

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(3.0, 1.0);
        
        assert!(limiter.try_acquire());
        assert!(limiter.try_acquire());
        assert!(limiter.try_acquire());
        assert!(!limiter.try_acquire()); // Bucket empty
    }

    #[test]
    fn test_message_priority() {
        assert!(MessagePriority::Urgent > MessagePriority::High);
        assert!(MessagePriority::High > MessagePriority::Normal);
        assert!(MessagePriority::Normal > MessagePriority::Low);
    }

    #[tokio::test]
    async fn test_queue_enqueue() {
        let queue = MessageQueue::default();
        let msg = OutgoingMessage {
            channel: ChannelId::new("test", "123"),
            content: crate::MessageContent::text("Hello"),
            reply_to: None,
        };

        let id = queue.enqueue(msg).await;
        assert_eq!(id, 0);
        assert_eq!(queue.len("test").await, 1);
    }

    /// Nobody reads the events: enqueueing past the event channel's 1000
    /// still returns. It used to block forever on the 1001st, holding the
    /// queues' write lock.
    #[tokio::test]
    async fn unread_events_never_block_the_queue() {
        let queue = MessageQueue::default();
        let enqueue_all = async {
            let mut last = 0;
            for n in 0..1_100 {
                last = queue
                    .enqueue(OutgoingMessage {
                        channel: ChannelId::new("test", "123"),
                        content: crate::MessageContent::text(format!("m{n}")),
                        reply_to: None,
                    })
                    .await;
            }
            last
        };
        let last = tokio::time::timeout(std::time::Duration::from_secs(10), enqueue_all)
            .await
            .expect("enqueue never blocks on unread events");
        assert_eq!(last, 1_099);
        assert!(queue.len("test").await > 0);
    }
}
