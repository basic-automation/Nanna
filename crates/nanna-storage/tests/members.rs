//! `members` table + [`MemberRepository`] (P25 Stage 1).
//!
//! The board treats the human and every agent as the same entity, so these
//! tests are mostly about that claim holding at the storage layer: one table,
//! one id space, and the seeded rows a fresh install needs before anyone has
//! configured anything.

#![warn(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]

use nanna_storage::{
    HUMAN_MEMBER_ID, MEMBER_AVATAR_MAX_BYTES, MEMBER_ID_MAX_BYTES, MEMBER_NAME_MAX_BYTES,
    MEMBER_PROFILE_MAX_BYTES, MemberKind, MemberOwner, MemberPatch, MemberStatus, NewMember,
    ROUTER_GLOBAL_MEMBER_ID, Storage, StorageError, WorkspaceRecord, router_member_id,
};

async fn storage() -> Storage {
    Storage::in_memory().await.expect("in-memory storage")
}

fn agent(id: &str) -> NewMember {
    NewMember {
        id: id.to_string(),
        name: "Test Agent".to_string(),
        avatar: None,
        kind: MemberKind::Agent,
        owner_kind: MemberOwner::Workspace,
        owner_id: None,
        status: MemberStatus::Idle,
        profile: serde_json::json!({}),
    }
}

/// The two rows a board cannot start without: somewhere to send a
/// clarification, and somebody to place a card.
#[tokio::test]
async fn a_fresh_database_seeds_the_human_and_the_global_router() {
    let storage = storage().await;
    let members = storage.members();

    let human = members.get(HUMAN_MEMBER_ID).await.expect("seeded human");
    assert_eq!(human.kind, MemberKind::Human);
    assert_eq!(human.owner_kind, MemberOwner::Human);
    assert_eq!(human.owner_id, None);
    assert_eq!(human.status, MemberStatus::Idle);

    let router = members
        .get(ROUTER_GLOBAL_MEMBER_ID)
        .await
        .expect("seeded global router");
    assert_eq!(router.kind, MemberKind::Agent);
    assert_eq!(router.owner_kind, MemberOwner::Workspace);
    assert_eq!(router.owner_id, None);
    assert_eq!(
        router
            .profile
            .get("role")
            .and_then(serde_json::Value::as_str),
        Some("router"),
        "the router announces itself in its profile"
    );

    assert_eq!(members.count().await.expect("count"), 2);
}

/// A workspace registered *before* the migration ran gets its router seeded by
/// the migration's `INSERT … SELECT`; one registered after gets it from
/// `ensure_router`. Both paths must produce the same id.
#[tokio::test]
async fn ensure_router_is_idempotent_and_scoped_to_the_workspace() {
    let storage = storage().await;
    storage
        .workspaces()
        .upsert(&WorkspaceRecord {
            id: "ws-a".to_string(),
            name: "A".to_string(),
            path: "/tmp/ws-a".to_string(),
            active: true,
            created_at: String::new(),
            last_accessed: String::new(),
        })
        .await
        .expect("register a workspace");

    let members = storage.members();
    let first = members
        .ensure_router(Some("ws-a"))
        .await
        .expect("create the router");
    assert_eq!(first.id, router_member_id(Some("ws-a")));
    assert_eq!(first.id, "router:ws-a");
    assert_eq!(first.owner_id.as_deref(), Some("ws-a"));

    let second = members
        .ensure_router(Some("ws-a"))
        .await
        .expect("return the existing router");
    assert_eq!(second.id, first.id);
    assert_eq!(second.created_at, first.created_at, "not re-created");

    // The global router is a different member, not this one under another name.
    let global = members
        .ensure_router(None)
        .await
        .expect("the global router already exists");
    assert_eq!(global.id, ROUTER_GLOBAL_MEMBER_ID);
    assert_ne!(global.id, first.id);
}

/// A workspace registered after the members migration ran still gets its
/// router: before `upsert` ensured one, only the migration's seed made routers,
/// so every board opened since had nobody to own its router profile.
#[tokio::test]
async fn registering_a_workspace_gives_its_board_a_router() {
    let storage = storage().await;
    let router = router_member_id(Some("ws-late"));
    assert!(
        !storage.members().exists(&router).await.expect("lookup"),
        "no router before the workspace exists"
    );
    let record = WorkspaceRecord {
        id: "ws-late".to_string(),
        name: "Late".to_string(),
        path: "/tmp/ws-late".to_string(),
        active: false,
        created_at: String::new(),
        last_accessed: String::new(),
    };
    storage.workspaces().upsert(&record).await.expect("register");
    let member = storage.members().get(&router).await.expect("the router exists");
    assert_eq!(member.owner_id.as_deref(), Some("ws-late"));
    assert_eq!(member.owner_kind, MemberOwner::Workspace);

    // Saving again (every activation re-saves the registry) is not a second router.
    storage.workspaces().upsert(&record).await.expect("re-save");
    let routers = storage
        .members()
        .list_for_workspace(Some("ws-late"))
        .await
        .expect("roster")
        .into_iter()
        .filter(|m| m.id == router)
        .count();
    assert_eq!(routers, 1);
}

/// A board sees its own workspace's members plus every human-owned one — a
/// personal agent travels with its owner (P25 decision 13).
#[tokio::test]
async fn a_board_sees_its_workspace_members_plus_human_owned_ones() {
    let storage = storage().await;
    let members = storage.members();

    members
        .create(NewMember {
            owner_id: Some("ws-a".to_string()),
            ..agent("agent-a")
        })
        .await
        .expect("a workspace-owned agent");
    members
        .create(NewMember {
            owner_id: Some("ws-b".to_string()),
            ..agent("agent-b")
        })
        .await
        .expect("another workspace's agent");
    members
        .create(NewMember {
            owner_kind: MemberOwner::Human,
            owner_id: Some(HUMAN_MEMBER_ID.to_string()),
            ..agent("agent-personal")
        })
        .await
        .expect("a personal agent");

    let mut ids: Vec<String> = members
        .list_for_workspace(Some("ws-a"))
        .await
        .expect("board roster")
        .into_iter()
        .map(|m| m.id)
        .collect();
    ids.sort();
    assert_eq!(
        ids,
        vec![
            "agent-a".to_string(),
            "agent-personal".to_string(),
            HUMAN_MEMBER_ID.to_string(),
        ],
        "the other workspace's agent is not on this board"
    );

    // The global board's workspace-owned members are the null-`owner_id` rows.
    let mut global: Vec<String> = members
        .list_for_workspace(None)
        .await
        .expect("global roster")
        .into_iter()
        .map(|m| m.id)
        .collect();
    global.sort();
    assert_eq!(
        global,
        vec![
            "agent-personal".to_string(),
            HUMAN_MEMBER_ID.to_string(),
            ROUTER_GLOBAL_MEMBER_ID.to_string(),
        ]
    );
}

#[tokio::test]
async fn an_id_can_only_be_taken_once() {
    let storage = storage().await;
    let members = storage.members();
    members.create(agent("dup")).await.expect("first create");
    let err = members
        .create(agent("dup"))
        .await
        .expect_err("a second member may not reuse the id");
    assert!(
        matches!(&err, StorageError::Invalid(message) if message.contains("already exists")),
        "{err:?}"
    );
}

#[tokio::test]
async fn update_applies_only_the_fields_it_carries() {
    let storage = storage().await;
    let members = storage.members();
    let before = members
        .create(NewMember {
            avatar: Some("🤖".to_string()),
            profile: serde_json::json!({ "tier": "small" }),
            ..agent("patchme")
        })
        .await
        .expect("create");

    let after = members
        .update(
            "patchme",
            MemberPatch {
                status: Some(MemberStatus::Busy),
                ..MemberPatch::default()
            },
        )
        .await
        .expect("patch the status only");

    assert_eq!(after.status, MemberStatus::Busy);
    assert_eq!(after.name, before.name, "name untouched");
    assert_eq!(after.avatar, before.avatar, "avatar untouched");
    assert_eq!(after.profile, before.profile, "profile untouched");

    // `Some(None)` clears; `None` leaves alone. The distinction is the whole
    // reason `avatar` is a double option.
    let cleared = members
        .update(
            "patchme",
            MemberPatch {
                avatar: Some(None),
                ..MemberPatch::default()
            },
        )
        .await
        .expect("clear the avatar");
    assert_eq!(cleared.avatar, None);
    assert_eq!(cleared.status, MemberStatus::Busy, "status untouched");
}

#[tokio::test]
async fn set_status_reports_a_member_that_does_not_exist() {
    let storage = storage().await;
    let members = storage.members();
    members
        .set_status(HUMAN_MEMBER_ID, MemberStatus::Busy)
        .await
        .expect("the human exists");
    assert_eq!(
        members
            .get(HUMAN_MEMBER_ID)
            .await
            .expect("read back")
            .status,
        MemberStatus::Busy
    );

    let err = members
        .set_status("nobody", MemberStatus::Busy)
        .await
        .expect_err("no such member");
    assert!(matches!(err, StorageError::NotFound(_)), "{err:?}");
}

/// Deleting the human or a router would leave the board with nowhere to send a
/// clarification or nobody to place a card.
#[tokio::test]
async fn the_human_and_the_routers_cannot_be_deleted() {
    let storage = storage().await;
    let members = storage.members();

    for protected in [HUMAN_MEMBER_ID, ROUTER_GLOBAL_MEMBER_ID] {
        let err = members
            .delete(protected)
            .await
            .expect_err("protected member");
        assert!(
            matches!(&err, StorageError::Invalid(message) if message.contains("cannot be deleted")),
            "{protected}: {err:?}"
        );
    }

    members.create(agent("expendable")).await.expect("create");
    assert!(members.delete("expendable").await.expect("delete"));
    assert!(
        !members.delete("expendable").await.expect("second delete"),
        "deleting an absent member is false, not an error"
    );
}

#[tokio::test]
async fn every_bound_is_enforced_on_create() {
    let storage = storage().await;
    let members = storage.members();

    let too_long_id = "x".repeat(MEMBER_ID_MAX_BYTES + 1);
    let cases: Vec<(&str, NewMember)> = vec![
        (
            "id",
            NewMember {
                id: too_long_id,
                ..agent("_")
            },
        ),
        (
            "id must not be empty",
            NewMember {
                id: String::new(),
                ..agent("_")
            },
        ),
        (
            "name",
            NewMember {
                name: "n".repeat(MEMBER_NAME_MAX_BYTES + 1),
                ..agent("bound-name")
            },
        ),
        (
            "name must not be empty",
            NewMember {
                name: "   ".to_string(),
                ..agent("bound-blank")
            },
        ),
        (
            "avatar",
            NewMember {
                avatar: Some("a".repeat(MEMBER_AVATAR_MAX_BYTES + 1)),
                ..agent("bound-avatar")
            },
        ),
        (
            "profile",
            NewMember {
                profile: serde_json::json!({ "pad": "p".repeat(MEMBER_PROFILE_MAX_BYTES) }),
                ..agent("bound-profile")
            },
        ),
    ];

    for (label, candidate) in cases {
        let err = match members.create(candidate).await {
            Ok(created) => panic!("{label}: created '{}' instead of rejecting it", created.id),
            Err(err) => err,
        };
        assert!(
            matches!(err, StorageError::Invalid(_)),
            "{label}: expected Invalid, got {err:?}"
        );
    }

    // Nothing above reached the table.
    assert_eq!(
        members.count().await.expect("count"),
        2,
        "only the two seeded members exist"
    );
}

/// An unknown token in `kind`/`owner_kind`/`status` is a row from a newer
/// schema. Coercing it to a default would silently change what a member *is*,
/// so the read fails instead.
#[tokio::test]
async fn an_unknown_enum_token_is_an_error_not_a_default() {
    let storage = storage().await;
    {
        let conn = storage.conn().lock().await;
        conn.execute(
            "INSERT INTO members (id, name, kind, owner_kind, owner_id, status, profile)
             VALUES ('from-the-future', 'X', 'daemon', 'workspace', NULL, 'idle', '{}')",
            (),
        )
        .await
        .expect("insert a row this version does not understand");
        drop(conn);
    }

    let err = storage
        .members()
        .get("from-the-future")
        .await
        .expect_err("an unknown kind must not decode");
    assert!(
        matches!(&err, StorageError::Invalid(message) if message.contains("unknown kind")),
        "{err:?}"
    );
}

/// The seeded human's id is the one thing `assignee = 'human'` relies on, and
/// the migration writes it directly — so assert the constant and the schema
/// agree rather than trusting that they do.
#[tokio::test]
async fn the_seeded_ids_match_the_constants_the_code_uses() {
    let storage = storage().await;
    let conn = storage.conn().lock().await;
    let mut rows = conn
        .query("SELECT id FROM members ORDER BY id ASC", ())
        .await
        .expect("read the seeded ids");
    let mut ids = Vec::new();
    while let Some(row) = rows.next().await.expect("a row") {
        ids.push(row.get::<String>(0).expect("an id"));
    }
    drop(rows);
    drop(conn);
    assert_eq!(
        ids,
        vec![
            HUMAN_MEMBER_ID.to_string(),
            ROUTER_GLOBAL_MEMBER_ID.to_string()
        ]
    );
}

/// `tasks.assignee` is a real reference to `members.id`. SQLite could not
/// express it as a constraint, so the check runs on write — this is the test
/// that says the reference exists at all.
#[tokio::test]
async fn a_task_can_only_be_assigned_to_a_member() {
    use nanna_storage::{NewTask, TaskPatch};

    let storage = storage().await;
    let tasks = storage.tasks();
    let members = storage.members();

    let err = tasks
        .create(NewTask {
            scope: "global".to_string(),
            title: "placed nowhere".to_string(),
            priority: 3,
            assignee: Some("nobody".to_string()),
            ..NewTask::default()
        })
        .await
        .expect_err("an assignee that is not a member");
    assert!(
        matches!(&err, StorageError::Invalid(message) if message.contains("not a board member")),
        "{err:?}"
    );

    // Unassigned is the state a card sits in before the router places it.
    let unplaced = tasks
        .create(NewTask {
            scope: "global".to_string(),
            title: "waiting on the router".to_string(),
            priority: 3,
            ..NewTask::default()
        })
        .await
        .expect("an unassigned card is fine");

    // And a real member is accepted, through create and through update alike.
    members
        .create(agent("worker"))
        .await
        .expect("create a member");
    let placed = tasks
        .create(NewTask {
            scope: "global".to_string(),
            title: "placed".to_string(),
            priority: 3,
            assignee: Some("worker".to_string()),
            ..NewTask::default()
        })
        .await
        .expect("a real member is a valid assignee");
    assert_eq!(placed.assignee.as_deref(), Some("worker"));

    let err = tasks
        .update(
            unplaced.id,
            TaskPatch {
                assignee: Some(Some("ghost".to_string())),
                ..TaskPatch::default()
            },
            None,
        )
        .await
        .expect_err("update is guarded too");
    assert!(
        matches!(&err, StorageError::Invalid(message) if message.contains("not a board member")),
        "{err:?}"
    );

    let reassigned = tasks
        .update(
            unplaced.id,
            TaskPatch {
                assignee: Some(Some(HUMAN_MEMBER_ID.to_string())),
                ..TaskPatch::default()
            },
            None,
        )
        .await
        .expect("assigning to the human");
    assert_eq!(reassigned.assignee.as_deref(), Some(HUMAN_MEMBER_ID));

    // Clearing an assignee is how a card goes back to the router.
    let cleared = tasks
        .update(
            unplaced.id,
            TaskPatch {
                assignee: Some(None),
                ..TaskPatch::default()
            },
            None,
        )
        .await
        .expect("clearing the assignee");
    assert_eq!(cleared.assignee, None);
}

/// A card's thread (P25 decision 2): every post names a member and says what
/// kind of post it is, and nothing can rewrite one.
#[tokio::test]
async fn a_thread_post_names_a_member_and_a_kind() {
    use nanna_storage::{NewTask, TaskNoteKind};

    let storage = storage().await;
    let tasks = storage.tasks();
    let card = tasks
        .create(NewTask {
            scope: "global".to_string(),
            title: "a card with a thread".to_string(),
            priority: 3,
            ..NewTask::default()
        })
        .await
        .expect("create the card");

    let asked = tasks
        .post(
            card.id,
            Some("gui"),
            Some(HUMAN_MEMBER_ID),
            TaskNoteKind::Question,
            "which environment?",
        )
        .await
        .expect("the human asks");
    assert_eq!(asked.kind, TaskNoteKind::Question);
    assert_eq!(asked.author_member_id.as_deref(), Some(HUMAN_MEMBER_ID));
    assert_eq!(asked.author.as_deref(), Some("gui"));

    // The author is a real reference, exactly like `tasks.assignee`.
    let err = tasks
        .post(
            card.id,
            None,
            Some("ghost"),
            TaskNoteKind::Comment,
            "from nowhere",
        )
        .await
        .expect_err("an author that is not a member");
    assert!(
        matches!(&err, StorageError::Invalid(message) if message.contains("note author")),
        "{err:?}"
    );

    // A post with no member is still allowed — that is what an inherited row
    // and a pre-members writer look like.
    let legacy = tasks
        .add_note(card.id, Some("harness"), "a finding")
        .await
        .expect("the legacy three-argument surface still works");
    assert_eq!(legacy.kind, TaskNoteKind::Comment, "it defaults to comment");
    assert_eq!(legacy.author_member_id, None);

    let thread = tasks.notes(card.id, 50).await.expect("read the thread");
    assert_eq!(thread.len(), 2, "oldest first, nothing dropped");
    assert_eq!(thread[0].id, asked.id);
    assert_eq!(thread[1].id, legacy.id);
}

/// An unknown kind is a post from a newer schema. Flattening it to a comment
/// would silently demote a question or a verdict to prose.
#[tokio::test]
async fn an_unknown_note_kind_is_an_error_not_a_comment() {
    use nanna_storage::NewTask;

    let storage = storage().await;
    let card = storage
        .tasks()
        .create(NewTask {
            scope: "global".to_string(),
            title: "future thread".to_string(),
            priority: 3,
            ..NewTask::default()
        })
        .await
        .expect("create");

    {
        let conn = storage.conn().lock().await;
        // `turso` is not a dev-dependency here, so this integration test cannot
        // build a params list — the id is an i64 this test just created, so
        // formatting it in is exact.
        conn.execute(
            &format!(
                "INSERT INTO task_notes (task_id, author, author_member_id, kind, content)
                 VALUES ({}, NULL, NULL, 'decree', 'from a newer schema')",
                card.id
            ),
            (),
        )
        .await
        .expect("insert a kind this version does not understand");
        drop(conn);
    }

    let err = storage
        .tasks()
        .notes(card.id, 50)
        .await
        .expect_err("an unknown kind must not decode");
    assert!(
        matches!(&err, StorageError::Invalid(message) if message.contains("unknown kind")),
        "{err:?}"
    );
}

/// What a freed member works next (P25 decision 3): its own open board cards,
/// best priority first then the oldest — never a card it has already picked
/// up, a closed one, another member's, or chat scaffolding.
#[tokio::test]
async fn a_members_waiting_cards_are_its_open_board_cards_best_first() {
    let storage = storage().await;
    storage
        .members()
        .create(agent("agent:a"))
        .await
        .expect("agent a");
    storage
        .members()
        .create(agent("agent:b"))
        .await
        .expect("agent b");
    let tasks = storage.tasks();
    let card = |title: &str, scope: &str, priority: i64, assignee: &str| nanna_storage::NewTask {
        title: title.to_string(),
        scope: scope.to_string(),
        scope_id: (scope == "session").then(|| "s1".to_string()),
        priority,
        assignee: Some(assignee.to_string()),
        ..nanna_storage::NewTask::default()
    };
    let low = tasks
        .create(card("low", "global", 4, "agent:a"))
        .await
        .expect("low");
    let urgent = tasks
        .create(card("urgent", "global", 1, "agent:a"))
        .await
        .expect("urgent");
    let also_low = tasks
        .create(card("also low", "global", 4, "agent:a"))
        .await
        .expect("also low");
    tasks
        .create(card("b's", "global", 1, "agent:b"))
        .await
        .expect("b's");
    tasks
        .create(card("chat", "session", 1, "agent:a"))
        .await
        .expect("chat");
    let started = tasks
        .create(card("started", "global", 1, "agent:a"))
        .await
        .expect("started");
    tasks
        .update(
            started.id,
            nanna_storage::TaskPatch {
                status: Some("in_progress".to_string()),
                ..nanna_storage::TaskPatch::default()
            },
            None,
        )
        .await
        .expect("picked up");
    let done = tasks
        .create(card("done", "global", 1, "agent:a"))
        .await
        .expect("done");
    tasks.complete(done.id, None, None).await.expect("closed");

    let waiting: Vec<i64> = tasks
        .open_board_cards_assigned_to("agent:a", 64)
        .await
        .expect("waiting")
        .iter()
        .map(|t| t.id)
        .collect();
    assert_eq!(waiting, vec![urgent.id, low.id, also_low.id]);
    let first_only = tasks
        .open_board_cards_assigned_to("agent:a", 1)
        .await
        .expect("one");
    assert_eq!(first_only.len(), 1, "the scan is bounded");
}

/// What a restarted daemon resumes: cards whose newest run marker is a start
/// — never one whose run ended, however it ended, and each card once.
#[tokio::test]
async fn a_card_whose_newest_run_marker_is_a_start_was_interrupted() {
    let storage = storage().await;
    let tasks = storage.tasks();
    let mut ids = Vec::new();
    for title in [
        "ended",
        "interrupted",
        "restarted then interrupted",
        "only ended",
    ] {
        let card = tasks
            .create(nanna_storage::NewTask {
                title: title.to_string(),
                scope: "global".to_string(),
                priority: 3,
                ..nanna_storage::NewTask::default()
            })
            .await
            .expect("card");
        ids.push(card.id);
    }
    let mark = |id: i64, action: &'static str| {
        let tasks = storage.tasks();
        async move {
            tasks
                .log_activity(id, None, action, None)
                .await
                .expect("marker");
        }
    };
    mark(ids[0], "run_started").await;
    mark(ids[1], "run_started").await;
    mark(ids[0], "run_ended").await;
    mark(ids[2], "run_started").await;
    mark(ids[2], "run_ended").await;
    mark(ids[3], "run_ended").await;
    mark(ids[2], "run_started").await;

    let unended = tasks
        .cards_with_unended("run_started", "run_ended", 1024)
        .await
        .expect("scan");
    assert_eq!(unended, vec![ids[2], ids[1]], "newest first, each once");
}

/// A card's assignee comes from the card's own board: the Inbox spans every
/// board, and a card from workspace B opened there offered workspace A's
/// roster — whose agents the store then accepted onto B's card.
#[tokio::test]
async fn a_card_is_assigned_only_to_a_member_of_its_own_board() {
    let storage = storage().await;
    let members = storage.members();
    for (id, owner) in [("agent:a-builder", "ws-a"), ("agent:b-builder", "ws-b")] {
        members
            .create(NewMember {
                owner_id: Some(owner.to_string()),
                ..agent(id)
            })
            .await
            .expect("member");
    }
    members.create(agent("agent:global")).await.expect("member");
    members
        .create(NewMember {
            owner_kind: MemberOwner::Human,
            owner_id: Some(HUMAN_MEMBER_ID.to_string()),
            ..agent("agent:personal")
        })
        .await
        .expect("member");

    let tasks = storage.tasks();
    let card = tasks
        .create(nanna_storage::NewTask {
            scope: "workspace".to_string(),
            scope_id: Some("ws-b".to_string()),
            title: "B's card".to_string(),
            priority: 3,
            ..nanna_storage::NewTask::default()
        })
        .await
        .expect("card");
    let assign = |who: &'static str| {
        let tasks = storage.tasks();
        async move {
            tasks
                .update(
                    card.id,
                    nanna_storage::TaskPatch {
                        assignee: Some(Some(who.to_string())),
                        ..nanna_storage::TaskPatch::default()
                    },
                    None,
                )
                .await
        }
    };
    for refused in ["agent:a-builder", "agent:global"] {
        let err = assign(refused).await;
        assert!(
            matches!(&err, Err(StorageError::Invalid(m)) if m.contains("not on this card's board")),
            "{refused}: {err:?}"
        );
    }
    for admitted in ["agent:b-builder", "agent:personal", HUMAN_MEMBER_ID] {
        assert!(assign(admitted).await.is_ok(), "{admitted} is on B's board");
    }
    let global_card = tasks
        .create(nanna_storage::NewTask {
            scope: "global".to_string(),
            title: "Global card".to_string(),
            priority: 3,
            assignee: Some("agent:b-builder".to_string()),
            ..nanna_storage::NewTask::default()
        })
        .await;
    assert!(
        global_card.is_err(),
        "a workspace agent is not on the global board"
    );
}
