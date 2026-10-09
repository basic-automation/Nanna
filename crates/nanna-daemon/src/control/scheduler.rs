//! Scheduler handlers for the [`ControlPlane`].

use super::{json, info, ControlPlane, RwLock, SchedulerAction, Value, Scheduler};

impl ControlPlane {
    // =========================================================================
    // Scheduler Handlers
    // =========================================================================
    
    pub(super) async fn handle_scheduler(
        &self,
        _client_id: &str,
        action: SchedulerAction,
    ) -> Value {
        let Some(ref scheduler) = self.scheduler else {
            return json!({ "error": "scheduler_unavailable", "message": "Scheduler not configured" });
        };
        
        match action {
            SchedulerAction::List => {
                let tasks = scheduler.read().await.list_tasks().await;
                let jobs: Vec<_> = tasks.iter().map(Self::scheduler_job_json).collect();
                json!({ "jobs": jobs })
            }
            SchedulerAction::Get { id } => {
                let task = scheduler.read().await.get_task(&id).await;
                task.as_ref().map_or_else(
                    || json!({ "error": "not_found", "id": id }),
                    |task| json!({ "job": Self::scheduler_job_json(task) }),
                )
            }
            SchedulerAction::Add {
                schedule,
                task,
                name,
                session_id,
            } => {
                self.scheduler_add(scheduler, schedule, task, name, session_id)
                    .await
            }
            SchedulerAction::Update { id, schedule, task, enabled } => {
                Self::scheduler_update(&*scheduler.read().await, id, schedule, task, enabled).await
            }
            SchedulerAction::Remove { id } => {
                let scheduler = scheduler.read().await;
                if scheduler.remove_task(&id).await {
                    info!("Removed scheduled job: {}", id);
                    json!({ "status": "deleted", "id": id })
                } else {
                    json!({ "error": "not_found", "id": id })
                }
            }
            SchedulerAction::RunNow { id } => {
                let scheduler = scheduler.read().await;
                match scheduler.run_now(&id).await {
                    Some(result) => {
                        json!({
                            "status": if result.success { "success" } else { "failed" },
                            "id": id,
                            "output": result.output,
                            "error": result.error,
                            "duration_ms": result.duration_ms,
                        })
                    }
                    None => json!({ "error": "not_found_or_no_executor", "id": id })
                }
            }
            SchedulerAction::History { id, limit } => {
                let runs = scheduler.read().await.get_history(&id, limit.unwrap_or(10)).await;
                let history: Vec<_> = runs.into_iter()
                    .map(|r| json!({
                        "run_id": r.id,
                        "job_id": r.job_id,
                        "started_at": r.started_at.to_rfc3339(),
                        "finished_at": r.finished_at.map(|dt| dt.to_rfc3339()),
                        "success": r.success,
                        "output": r.output,
                        "error": r.error,
                    }))
                    .collect();
                json!({ "history": history, "job_id": id })
            }
        }
    }

    /// `SchedulerAction::Update`: change a job's schedule, payload and/or
    /// enabled flag.
    ///
    /// An unknown id is `not_found` whatever was asked. It used to answer
    /// `updated` whenever no schedule was given (enabling a job that does not
    /// exist "succeeded"), and a new payload was accepted and silently dropped.
    async fn scheduler_update(
        scheduler: &nanna_core::Scheduler,
        id: String,
        schedule: Option<String>,
        task: Option<String>,
        enabled: Option<bool>,
    ) -> Value {
        if scheduler.get_task(&id).await.is_none() {
            return json!({ "error": "not_found", "id": id });
        }
        if let Some(new_schedule) = schedule {
            match scheduler.update_schedule(&id, &new_schedule).await {
                Ok(true) => {}
                Ok(false) => return json!({ "error": "not_found", "id": id }),
                Err(e) => {
                    return json!({ "error": "invalid_schedule", "message": e.to_string() });
                }
            }
        }
        if let Some(payload) = task
            && !scheduler.update_payload(&id, &payload).await
        {
            return json!({ "error": "not_found", "id": id });
        }
        if let Some(en) = enabled
            && !scheduler.set_task_enabled(&id, en).await
        {
            return json!({ "error": "not_found", "id": id });
        }
        json!({ "status": "updated", "id": id })
    }

    /// `SchedulerAction::Add`: parse `schedule` as cron and store the job.
    async fn scheduler_add(
        &self,
        scheduler: &RwLock<Scheduler>,
        schedule: String,
        task: String,
        name: Option<String>,
        session_id: Option<String>,
    ) -> Value {
        // A result posted into a conversation that does not exist would
        // be dropped on every run; refuse it now instead.
        if let Some(ref id) = session_id
            && !self.sessions.exists(id).await
        {
            return json!({
                "error": "session_not_found",
                "message": format!("Session {id} not found; the job was not added"),
            });
        }
        // Try to parse as cron expression
        match Scheduler::cron_task(name.as_deref().unwrap_or("unnamed"), &schedule, &task) {
            Ok(mut scheduled_task) => {
                scheduled_task.target_session = session_id;
                let id = scheduled_task.id.clone();
                scheduler.read().await.add_task(scheduled_task).await;
                info!("Added scheduled job: {}", id);
                json!({ "status": "created", "id": id })
            }
            Err(e) => {
                json!({ "error": "invalid_schedule", "message": e.to_string() })
            }
        }
    }

    /// One scheduled job as the `list` and `get` replies render it.
    fn scheduler_job_json(task: &nanna_core::ScheduledTask) -> Value {
        let schedule = task.task_type.schedule_label();
        let next_run = task.task_type.next_run().map(|dt| dt.to_rfc3339());
        json!({
            "id": task.id,
            "name": task.name,
            "schedule": schedule,
            "payload": task.payload,
            "enabled": task.enabled,
            "last_run": task.last_run.map(|dt| dt.to_rfc3339()),
            "next_run": next_run,
            "run_count": task.run_count,
            "timezone": task.timezone,
            "target_channel": task.target_channel,
            "target_session": task.target_session,
        })
    }
}
