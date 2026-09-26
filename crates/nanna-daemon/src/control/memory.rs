//! Memory handlers for the [`ControlPlane`].

use super::{json, warn, info, error, ControlPlane, MemoryAction, MemoryService, Value, ConsolidationConfig};

impl ControlPlane {
    // =========================================================================
    // Memory Handlers
    // =========================================================================
    
    /// The wire `scope` as a [`nanna_memory::RecallScope`]: `None` = every
    /// memory, `"global"` = global memories only, a workspace id = global
    /// memories plus that workspace's. One parse for `search`, `list` and
    /// `export`, so they can never disagree about what a scope contains —
    /// `search` used to read `"global"` as *every* memory while the other two
    /// read it as the global ones.
    fn memory_scope(scope: Option<&str>) -> nanna_memory::RecallScope<'_> {
        match scope {
            None => nanna_memory::RecallScope::Everything,
            Some("global") => nanna_memory::RecallScope::GlobalOnly,
            Some(workspace) => nanna_memory::RecallScope::Workspace(workspace),
        }
    }

    pub(super) async fn handle_memory(&self, _client_id: &str, action: MemoryAction) -> Value {
        let Some(ref memory) = self.memory else {
            return json!({ "error": "memory_unavailable", "message": "Memory service not configured" });
        };

        match action {
            MemoryAction::List { scope } => Self::memory_list(memory, scope).await,
            MemoryAction::Search { query, limit, scope } => Self::memory_search(memory, query, limit, scope).await,
            MemoryAction::Get { id } => {
                // Get memory by ID
                if let Some(entry) = memory.get(&id).await {
                    // Absent provenance is "unknown", never "stated" (see `memory_list`).
                    let fact_type = entry.metadata.get("fact_type")
                        .cloned()
                        .unwrap_or_else(|| "unknown".to_string());
                    let created_at = chrono::DateTime::from_timestamp(entry.timestamp, 0).map_or_else(|| entry.timestamp.to_string(), |dt| dt.format("%Y-%m-%d %H:%M:%S").to_string());
                    
                    json!({
                        "memory": {
                            "id": entry.id,
                            "content": entry.content,
                            "fact_type": fact_type,
                            "importance": entry.importance,
                            "state": format!("{:?}", entry.state).to_lowercase(),
                            "weight": entry.weight,
                            "retrievability": entry.retrievability,
                            "access_count": entry.access_count,
                            "created_at": created_at,
                            "session_id": entry.metadata.get("session_id"),
                            "workspace_id": entry.workspace_id,
                        }
                    })
                } else {
                    json!({ "error": "not_found", "id": id })
                }
            }
            MemoryAction::Create { content, tags, importance } => {
                let mut metadata = std::collections::HashMap::new();
                if let Some(tags) = tags {
                    metadata.insert("tags".to_string(), tags.join(","));
                }

                match memory.remember_with_importance(&content, metadata, f32::from(importance.unwrap_or(3))).await {
                    Ok((id, action)) => {
                        // Memory auto-persisted to Turso via write-through.
                        json!({
                            "id": id,
                            "action": format!("{:?}", action),
                        })
                    }
                    Err(e) => json!({ "error": "create_failed", "message": e.to_string() })
                }
            }
            MemoryAction::Update { id, content, tags: _ } => {
                // Update memory content
                if let Some(new_content) = content {
                    match memory.update_content(&id, &new_content).await {
                        Ok(()) => {
                            // Memory auto-persisted to Turso via write-through.
                            json!({ "status": "updated", "id": id })
                        }
                        Err(e) => json!({ "error": "update_failed", "message": e.to_string() })
                    }
                } else {
                    json!({ "error": "no_changes", "id": id })
                }
            }
            MemoryAction::Delete { id } => {
                match memory.forget(&id).await {
                    Ok(()) => {
                        // Memory auto-persisted to Turso via write-through.
                        json!({ "status": "deleted", "id": id })
                    }
                    Err(e) => json!({ "error": "delete_failed", "message": e.to_string() })
                }
            }
            MemoryAction::Clear { scope } => Self::memory_clear(memory, scope).await,
            MemoryAction::Export { scope, format } => Self::memory_export(memory, scope, format).await,
            MemoryAction::Stats => {
                let stats = memory.stats().await;
                json!({
                    "total": stats.total,
                    "active": stats.active,
                    "dormant": stats.dormant,
                    "silent": stats.silent,
                    "unavailable": stats.unavailable,
                    // How long memories written without a vector stayed
                    // unfindable by similarity search; null until one has
                    // been filled this run.
                    "queue_to_searchable": stats.searchable_latency.map(|l| json!({
                        "samples": l.samples,
                        "p50_secs": l.p50_secs,
                        "p95_secs": l.p95_secs,
                    })),
                })
            }
            MemoryAction::Consolidate => self.memory_consolidate(memory).await,
        }
    }

    /// `MemoryAction::Search`: scoped recall, saying how much of the store was searchable.
    async fn memory_search(
        memory: &MemoryService,
        query: String,
        limit: Option<usize>,
        scope: Option<String>,
    ) -> Value {
        let result = memory
            .recall_in_scope_with_report(&query, Self::memory_scope(scope.as_deref()))
            .await;
        match result {
            Ok(nanna_memory::RecallReport {
                results,
                coverage,
                timings,
            }) => {
                let memories: Vec<_> = results.into_iter()
                    .take(limit.unwrap_or(10))
                    .map(|r| json!({
                        "id": r.id,
                        "content": r.content,
                        "score": r.score,
                        "weight": r.weight,
                    }))
                    .collect();
                // An empty answer from a scan that could not read the
                // whole store is NOT "nothing matched". Saying which of
                // the three states produced it is the difference
                // between a miss the caller can act on and a blackout
                // reported as a clean result.
                let note = if !memories.is_empty() {
                    None
                } else if !coverage.query_width_matches {
                    Some(format!(
                        "No memories were searched: this query's embedding does not \
                         match the store's current binding, so all {} memories are \
                         unreachable until the store is re-bound or backfilled.",
                        coverage.total
                    ))
                } else if coverage.unsearchable() > 0 {
                    Some(format!(
                        "Nothing matched among the {} of {} memories that could be \
                         searched; the other {} are awaiting re-embedding and were not \
                         consulted.",
                        coverage.comparable,
                        coverage.total,
                        coverage.unsearchable()
                    ))
                } else {
                    None
                };
                json!({
                    "memories": memories,
                    "query": query,
                    "searched": coverage.comparable,
                    "total": coverage.total,
                    "note": note,
                    // Per stage, because they are not alike: the embed
                    // is a model call, the search an in-RAM scan.
                    "timings": {
                        "embed_ms": timings.embed_ms(),
                        "search_ms": timings.search_ms(),
                    },
                })
            }
            Err(e) => json!({ "error": "search_failed", "message": e.to_string() })
        }
    }

    /// `MemoryAction::Clear`: every memory, or one scope's — durably either way.
    ///
    /// `None` used to empty RAM only and answer `"cleared"`, so the next restart
    /// loaded every memory back from Turso. Both arms now go through
    /// [`MemoryService::forget_many`], which removes a memory from RAM only once
    /// its row is gone; a partial failure is reported as one, never as success.
    ///
    /// Destructive scopes stay narrow, unlike `list`'s: `"global"` clears only
    /// unscoped memories, and a workspace id clears ONLY that workspace's —
    /// never the globals its tab also displays.
    async fn memory_clear(memory: &MemoryService, scope: Option<String>) -> Value {
        let entries = memory.list_all().await;
        let ids: Vec<&str> = entries
            .iter()
            .filter(|m| match scope.as_deref() {
                None => true,
                Some("global") => m.workspace_id.is_none(),
                Some(ws) => m.workspace_id.as_deref() == Some(ws),
            })
            .map(|m| m.id.as_str())
            .collect();
        let outcome = memory.forget_many(&ids).await;
        let scope_name = scope.as_deref().unwrap_or("all");
        info!(
            "Cleared {} of {} memories in scope {} ({} refused)",
            outcome.removed,
            ids.len(),
            scope_name,
            outcome.failed
        );
        if outcome.failed > 0 {
            return json!({
                "error": "clear_incomplete",
                "scope": scope_name,
                "removed": outcome.removed,
                "failed": outcome.failed,
                "message": format!(
                    "{} memories could not be deleted from storage and were kept: {}",
                    outcome.failed,
                    outcome.first_error.unwrap_or_default()
                ),
            });
        }
        json!({ "status": "cleared", "scope": scope_name, "removed": outcome.removed })
    }

    /// `MemoryAction::Consolidate`: one explicit, non-idle-gated dream cycle.
    async fn memory_consolidate(&self, memory: &MemoryService) -> Value {
        // Route through the daemon's single dreaming orchestrator so a
        // manual consolidation runs the *same* body as the scheduled
        // one: apply tallied feedback, flush the pending FSRS
        // testing-effect queue that every `recall` fills, then
        // consolidate. Calling `MemoryService::consolidate` directly
        // (the old shape) skipped both and left that queue to grow for
        // the daemon's whole uptime.
        //
        // Falls back to the low-level `MemoryService::consolidate` when
        // memory is on but no orchestrator is attached (minimal
        // constructions), so a manual consolidation can never regress.

        // Trigger memory consolidation (requires LLM for summarization)
        let Some(ref router) = self.router else {
            return json!({ "error": "llm_unavailable", "message": "LLM router required for consolidation" });
        };

        let router_for_summarize = router.clone();

        // Summarization models from settings, falling back to the chat
        // models in their configured order. Shared with the scheduled dream
        // cycle and `memory.summarize` so the three cannot drift (see
        // `crate::dream_summarizer`).
        let cfg = self.config.read().await;
        let summarize_models = crate::dream_summarizer::for_llm_config(&cfg.llm);
        let max_compression_ratio = cfg.memory.max_compression_ratio;
        let min_remaining_memories = cfg.memory.min_remaining_memories;
        drop(cfg);

        if summarize_models.is_empty() {
            return json!({ "error": "no_models", "message": "No summarization or main models configured." });
        }

        // Size the budget to the SMALLEST window across the failover
        // list — one prompt is offered to each candidate in turn, so a
        // budget fitted to the first would overflow a smaller fallback.
        let window_tokens = crate::dream_summarizer::summarizer_context_window_tokens(
            router,
            &summarize_models,
        )
        .await;
        let config = ConsolidationConfig {
            max_compression_ratio,
            min_remaining_memories,
            ..ConsolidationConfig::default()
        }
        .with_summarizer_context_window(window_tokens);

        info!("Consolidation summarization priority: {:?}", summarize_models);

        let summarize = crate::dream_summarizer::summarize_with_failover(
            router_for_summarize,
            summarize_models,
        );

        // Deliberately NOT idle-gated: the user asked for this one
        // explicitly, so it runs even while the system is busy.
        //
        // P13 unification: dream through the shared `DreamingService`
        // when one is attached, so a user-triggered consolidation runs
        // the full multi-phase cycle — pending feedback applied, the
        // testing-effect FSRS flush, the min-memories floor — and not
        // just its final clustering phase. Falls back to the low-level
        // call when no orchestrator is attached (minimal
        // constructions); the fallback reports zero promotions and
        // demotions because only the orchestrator tallies feedback.
        let dreamed = if let Some(ref dreaming) = self.dreaming {
            dreaming.dream_with_consolidation(&config, summarize).await
        } else {
            let fallback = memory.consolidate(&config, summarize).await;
            let total_memories = memory.count().await;
            fallback.map(|consolidation| nanna_memory::DreamingStats {
                consolidation,
                auto_promoted: 0,
                auto_demoted: 0,
                total_memories,
            })
        };

        match dreamed {
            Ok(stats) => {
                let result = stats.consolidation;
                info!(
                    "Memory consolidation: {} processed, {} clusters, {} merged, {} expanded, {} promoted, {} demoted, {} errors",
                    result.memories_processed,
                    result.clusters_formed,
                    result.memories_merged,
                    result.memories_expanded,
                    stats.auto_promoted,
                    stats.auto_demoted,
                    result.errors.len()
                );
                for err in &result.errors {
                    warn!("Consolidation error: {}", err);
                }
                json!({
                    "status": "success",
                    "memories_processed": result.memories_processed,
                    "clusters_formed": result.clusters_formed,
                    "memories_merged": result.memories_merged,
                    "memories_deduped": result.memories_deduped,
                    "memories_expanded": result.memories_expanded,
                    "auto_promoted": stats.auto_promoted,
                    "auto_demoted": stats.auto_demoted,
                    "total_memories": stats.total_memories,
                    "errors": result.errors,
                })
            }
            Err(e) => {
                let err_msg = e.to_string();
                error!("Memory consolidation failed: {}", err_msg);
                json!({ "error": "consolidation_failed", "message": err_msg })
            }
        }
    }

    /// `MemoryAction::List`: every memory in `scope`, newest state included.
    async fn memory_list(memory: &MemoryService, scope: Option<String>) -> Value {
        let all_memories = memory.list_all().await;
        let memories: Vec<_> = all_memories.into_iter()
            .filter(|m| Self::memory_scope(scope.as_deref()).admits(m.workspace_id.as_deref()))
            .map(|m| {
                // Absent provenance is "unknown" — NOT "stated". A legacy
                // memory stored before provenance was captured must not be
                // able to impersonate a user assertion.
                let fact_type = m.metadata.get("fact_type")
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string());
                let created_at = chrono::DateTime::from_timestamp(m.timestamp, 0).map_or_else(|| m.timestamp.to_string(), |dt| dt.format("%Y-%m-%d %H:%M:%S").to_string());

                json!({
                    "id": m.id,
                    "content": m.content,
                    "fact_type": fact_type,
                    "importance": m.importance,
                    "state": format!("{:?}", m.state).to_lowercase(),
                    "weight": m.weight,
                    "retrievability": m.retrievability,
                    "access_count": m.access_count,
                    "created_at": created_at,
                    "session_id": m.metadata.get("session_id"),
                    "workspace_id": m.workspace_id,
                })
            })
            .collect();
        json!({ "memories": memories })
    }

    /// `MemoryAction::Export`: render the scope's memories as a document.
    async fn memory_export(
        memory: &MemoryService,
        scope: Option<String>,
        format: crate::protocol::ExportFormat,
    ) -> Value {
        // Rendered here for the same reason as `session.export`: the
        // store's owner renders once and every client gets that
        // document. Filtered by the same rule `list` uses.
        let records: Vec<_> = memory
            .export_records()
            .await
            .into_iter()
            .filter(|m| Self::memory_scope(scope.as_deref()).admits(m.workspace_id.as_deref()))
            .collect();
        match crate::export::export_memories(
            &records,
            scope.as_deref(),
            format,
            chrono::Utc::now(),
        ) {
            Ok(document) => json!({
                "format": format,
                "filename": document.filename,
                "content": document.content,
                "count": records.len(),
            }),
            Err(e) => json!({
                "error": "export_failed",
                "message": format!("Memories could not be exported: {e}"),
            }),
        }
    }
}
