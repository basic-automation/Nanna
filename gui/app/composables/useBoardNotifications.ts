import { listen, type UnlistenFn } from '@tauri-apps/api/event'
import { invoke } from '@tauri-apps/api/core'
import { boardNoticeFor, type BoardCard, type BoardEvent } from '~/lib/board'
import { useNotificationCenter } from '~/composables/useNotificationCenter'

/** The event kinds that can notify; every other card change is the board's to show. */
const NOTIFYING_KINDS = new Set(['due', 'overdue', 'created', 'assigned'])

/**
 * Turn the board's own lifecycle events into Notification Center entries for
 * the human (see `boardNoticeFor`). Mounted once by the layout, so it hears
 * them on every page, not only while the board is open. Returns the unlisten.
 */
export async function startBoardNotifications(): Promise<UnlistenFn> {
  const { addNotification } = useNotificationCenter()
  return listen<BoardEvent>('board-event', async ({ payload }) => {
    if (!payload || !NOTIFYING_KINDS.has(payload.kind)) return
    try {
      const reply = await invoke<{ task?: BoardCard, error?: string }>('get_card', { id: payload.task_id })
      if (!reply?.task) return
      const notice = boardNoticeFor(payload, reply.task)
      if (!notice) return
      addNotification({
        type: notice.type,
        title: notice.title,
        summary: notice.summary,
        detail: `Card #${reply.task.id} — open the Board to see it.`,
        source: 'board',
        metadata: { taskId: reply.task.id },
      })
    } catch (e) {
      console.error('Board notification failed:', e)
    }
  })
}
