//! Board member handlers for the [`ControlPlane`] (P25 decision 3).
//!
//! Until this existed the roster was only what the members migration seeded —
//! the human and one router per board — so the router could assign a card to
//! nobody but the human, and a router's own `profile.model_priority` could not
//! be set from anywhere.

use super::{ControlPlane, Event, MemberAction, Value, json};
use nanna_storage::{
    HUMAN_MEMBER_ID, MEMBER_ID_MAX_BYTES, MemberKind, MemberOwner, MemberPatch, MemberRepository,
    MemberStatus, NewMember, StorageError,
};

// Prefix of every member created over IPC — disjoint from the seeded ids
// (`human`, `router:…`), so a created agent can never shadow a member the
// board depends on. Shared with the router's hand-back wake, which tells an
// agent's own writes by it.
use crate::board_router_trigger::AGENT_MEMBER_PREFIX;

/// Bytes of the id left for the slug once the prefix is in.
const AGENT_SLUG_MAX_BYTES: usize = MEMBER_ID_MAX_BYTES - AGENT_MEMBER_PREFIX.len();

/// Bytes of a random suffix used when a name has no ASCII letter or digit to
/// derive an id from (`"ミク"`, `"🤖"`): 8 hex digits of a v4 uuid — 2^32
/// values, so two such agents colliding is a refused create, not a silent
/// overwrite.
const AGENT_RANDOM_SLUG_BYTES: usize = 8;

impl ControlPlane {
    pub(super) async fn handle_member(&self, action: MemberAction) -> Value {
        let Some(ref storage) = self.storage else {
            return json!({"error": "storage_unavailable", "message": "the board roster requires storage"});
        };
        let repo = storage.members();
        let mutates = !matches!(action, MemberAction::List { .. } | MemberAction::Get { .. });
        let response = match action {
            MemberAction::List { workspace_id } => {
                match repo.list_for_workspace(workspace_id.as_deref()).await {
                    Ok(members) => json!({ "members": members }),
                    Err(e) => storage_error("member_list_failed", &e),
                }
            }
            MemberAction::Get { id } => match repo.get(&id).await {
                Ok(member) => json!({ "member": member }),
                Err(e) => storage_error("member_get_failed", &e),
            },
            MemberAction::Create {
                name,
                workspace_id,
                personal,
                avatar,
                profile,
            } => {
                let request = CreateMember {
                    name,
                    workspace_id,
                    personal,
                    avatar,
                    profile,
                };
                self.member_create(&repo, request).await
            }
            MemberAction::Update {
                id,
                name,
                avatar,
                status,
                profile,
            } => member_update(&repo, &id, name, avatar, status.as_deref(), profile).await,
            MemberAction::Delete { id } => match repo.delete(&id).await {
                Ok(removed) => json!({ "removed": removed }),
                Err(e) => storage_error("member_delete_failed", &e),
            },
        };
        if mutates && roster_changed(&response) {
            self.notify_members_changed();
        }
        response
    }

    /// Tell every connected client the roster changed. Fire-and-forget: a
    /// send error only means nobody is subscribed.
    fn notify_members_changed(&self) {
        if let Some(ref tx) = self.event_tx {
            let _ = tx.send(Event::MembersChanged);
        }
    }

    async fn member_create(&self, repo: &MemberRepository, request: CreateMember) -> Value {
        let CreateMember {
            name,
            workspace_id,
            personal,
            avatar,
            profile,
        } = request;
        if personal && workspace_id.is_some() {
            return invalid(
                "a personal agent belongs to the human and travels between boards; send \
                 either \"personal\": true or a \"workspace_id\", not both",
            );
        }
        if let Some(ref id) = workspace_id
            && self.workspaces.read().await.get(id).is_none()
        {
            return invalid(&format!(
                "workspace '{id}' is not registered — open it first, or omit \"workspace_id\" \
                 for the global board"
            ));
        }
        let profile = match admit_profile(profile) {
            Ok(profile) => profile,
            Err(why) => return invalid(&why),
        };
        let (owner_kind, owner_id) = if personal {
            (MemberOwner::Human, Some(HUMAN_MEMBER_ID.to_string()))
        } else {
            (MemberOwner::Workspace, workspace_id)
        };
        let new = NewMember {
            id: agent_member_id(&name),
            name,
            avatar,
            kind: MemberKind::Agent,
            owner_kind,
            owner_id,
            status: MemberStatus::Idle,
            profile,
        };
        debug_assert!(
            new.id.starts_with(AGENT_MEMBER_PREFIX),
            "created members are agents"
        );
        match repo.create(new).await {
            Ok(member) => json!({ "member": member }),
            Err(e) => storage_error("member_create_failed", &e),
        }
    }
}

/// The fields of [`MemberAction::Create`], so the handler stays one call.
struct CreateMember {
    name: String,
    workspace_id: Option<String>,
    personal: bool,
    avatar: Option<String>,
    profile: Option<Value>,
}

async fn member_update(
    repo: &MemberRepository,
    id: &str,
    name: Option<String>,
    avatar: Option<String>,
    status: Option<&str>,
    profile: Option<Value>,
) -> Value {
    let status = match status.map(|token| (token, MemberStatus::parse(token))) {
        None => None,
        Some((_, Some(status))) => Some(status),
        Some((token, None)) => {
            return invalid(&format!(
                "unknown status '{token}' — a member is \"idle\", \"busy\" or \"offline\""
            ));
        }
    };
    let profile = match profile.map(|p| admit_profile(Some(p))).transpose() {
        Ok(profile) => profile,
        Err(why) => return invalid(&why),
    };
    let patch = MemberPatch {
        name,
        avatar: avatar.map(Some),
        // Ownership is identity, like `kind`: not editable over IPC.
        owner_id: None,
        status,
        profile,
    };
    match repo.update(id, patch).await {
        Ok(member) => json!({ "member": member }),
        Err(e) => storage_error("member_update_failed", &e),
    }
}

/// A profile is a JSON object or nothing: the router reads named fields off it
/// (`model_priority`, capability tags), and a bare string or array would make
/// every one of those reads silently come back empty.
fn admit_profile(profile: Option<Value>) -> Result<Value, String> {
    match profile {
        None | Some(Value::Null) => Ok(json!({})),
        Some(Value::Object(fields)) => Ok(Value::Object(fields)),
        Some(other) => Err(format!(
            "a member profile is a JSON object of named fields, not {}",
            json_kind(&other)
        )),
    }
}

const fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// `agent:<slug of name>`: lowercase ASCII letters and digits joined by single
/// dashes, cut to fit [`MEMBER_ID_MAX_BYTES`]. A name with nothing ASCII to
/// slug gets a random suffix instead of being refused.
///
/// Derived from the name so the id reads as who it is on every card and in the
/// router's prompt; two agents with the same slug are a refused create
/// ("already exists"), never a merge.
pub(super) fn agent_member_id(name: &str) -> String {
    let mut slug = String::with_capacity(AGENT_SLUG_MAX_BYTES);
    for ch in name.chars() {
        if slug.len() >= AGENT_SLUG_MAX_BYTES {
            break;
        }
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
        } else if !slug.is_empty() && !slug.ends_with('-') {
            slug.push('-');
        }
    }
    let mut slug = slug.trim_end_matches('-').to_string();
    if slug.is_empty() {
        slug = uuid::Uuid::new_v4().simple().to_string();
        slug.truncate(AGENT_RANDOM_SLUG_BYTES);
    }
    let id = format!("{AGENT_MEMBER_PREFIX}{slug}");
    debug_assert!(
        id.len() <= MEMBER_ID_MAX_BYTES,
        "the slug is cut to fit the id bound"
    );
    debug_assert!(
        slug.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-'),
        "the slug is ASCII letters, digits and dashes"
    );
    id
}

/// Whether a mutating action's response says the roster changed: a member
/// came back (create, update) or one was removed. A refusal or a delete of an
/// id that did not exist changed nothing, and announcing it would send every
/// client to re-list for no reason.
fn roster_changed(response: &Value) -> bool {
    response.get("error").is_none()
        && (response.get("member").is_some() || response["removed"] == true)
}

fn invalid(message: &str) -> Value {
    json!({ "error": "invalid_member", "message": message })
}

/// A refused write (a bound, a protected member, a duplicate) is the caller's
/// to fix and says so; anything else is the store's.
fn storage_error(code: &str, error: &StorageError) -> Value {
    match error {
        StorageError::Invalid(why) => invalid(why),
        StorageError::NotFound(what) => json!({ "error": "member_not_found", "message": what }),
        other => json!({ "error": code, "message": other.to_string() }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_agent_id_is_the_slug_of_its_name() {
        assert_eq!(agent_member_id("Ada Lovelace"), "agent:ada-lovelace");
        assert_eq!(
            agent_member_id("  Code--Reviewer #2 "),
            "agent:code-reviewer-2"
        );
        assert_eq!(agent_member_id("Ada 🤖"), "agent:ada");
    }

    #[test]
    fn a_name_with_nothing_to_slug_gets_a_random_id() {
        let id = agent_member_id("ミク");
        assert!(id.starts_with(AGENT_MEMBER_PREFIX), "{id}");
        assert_eq!(
            id.len(),
            AGENT_MEMBER_PREFIX.len() + AGENT_RANDOM_SLUG_BYTES,
            "{id}"
        );
        assert_ne!(
            id,
            agent_member_id("ミク"),
            "two such agents do not collide"
        );
    }

    #[test]
    fn a_long_name_is_cut_to_the_id_bound() {
        let id = agent_member_id(&"a".repeat(500));
        assert_eq!(id.len(), MEMBER_ID_MAX_BYTES);
    }

    #[test]
    fn a_profile_is_an_object_or_nothing() {
        assert_eq!(admit_profile(None), Ok(json!({})));
        assert_eq!(admit_profile(Some(Value::Null)), Ok(json!({})));
        let object = json!({ "model_priority": ["m"] });
        assert_eq!(admit_profile(Some(object.clone())), Ok(object));
        let refused = admit_profile(Some(json!(["m"]))).unwrap_err();
        assert!(refused.contains("an array"), "{refused}");
    }
}
