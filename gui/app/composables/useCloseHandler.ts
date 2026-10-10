import { invoke } from '@tauri-apps/api/core'
import { getCurrentWindow } from '@tauri-apps/api/window'
import { exit } from '@tauri-apps/plugin-process'
import { readonly, ref } from 'vue'

export type CloseMode = 'ask' | 'minimize_to_tray' | 'quit_completely'

/**
 * Where a remembered close choice survives a restart. The daemon-side value
 * (`set_close_mode`) lives in the GUI process only and starts as "ask" every
 * launch, so "Remember my choice" was forgotten at the next start (and a
 * remembered "quit" died with the process at once). The browser store keeps
 * it and is pushed back on load. Reads and writes are best-effort.
 */
const CLOSE_MODE_KEY = 'nanna.closeMode'

function storedCloseMode(): CloseMode | null {
  try {
    const value = localStorage.getItem(CLOSE_MODE_KEY)
    return value === 'minimize_to_tray' || value === 'quit_completely' ? value : null
  } catch {
    return null
  }
}

const showCloseDialog = ref(false)
const closeMode = ref<CloseMode>('ask')
const rememberChoice = ref(false)

export function useCloseHandler() {
  /**
   * Load the saved close mode preference
   */
  async function loadCloseMode(): Promise<CloseMode> {
    try {
      const mode = await invoke<string>('get_close_mode')
      const remembered = storedCloseMode()
      if (mode === 'ask' && remembered) {
        await invoke('set_close_mode', { mode: remembered })
        closeMode.value = remembered
        return remembered
      }
      closeMode.value = mode as CloseMode
      return mode as CloseMode
    } catch (e) {
      console.error('Failed to load close mode:', e)
      return 'ask'
    }
  }
  
  /**
   * Save the close mode preference
   */
  async function setCloseMode(mode: CloseMode): Promise<void> {
    try {
      await invoke('set_close_mode', { mode })
      closeMode.value = mode
      try {
        localStorage.setItem(CLOSE_MODE_KEY, mode)
      } catch {
        // Not persisted; this session still uses it.
      }
    } catch (e) {
      console.error('Failed to set close mode:', e)
    }
  }
  
  /**
   * Act on a request to close the window, as the saved close mode says: open
   * the close dialog ("ask"), hide to the tray, or quit. It does all of it
   * itself, so a caller has nothing left to do afterwards.
   *
   * The window's own close is always prevented by its caller (the JS API
   * would otherwise destroy the window, which the capabilities do not
   * allow), so a close that did nothing here would leave no way out of the
   * window. When the close mode cannot be read, this quits.
   */
  async function handleClose(): Promise<void> {
    let action: string
    try {
      action = await invoke<string>('handle_window_close')
    } catch (e) {
      // Refused until Rust has managed the app's state, which waits on
      // Config::load's keyring read (a locked Secret Service holds it for as
      // long as its unlock prompt is up).
      console.error('[useCloseHandler] Close mode unavailable, quitting:', e)
      await performQuit()
      return
    }

    switch (action) {
      case 'ask':
        showCloseDialog.value = true
        return
      case 'minimized':
        // handle_window_close already hid the window.
        return
      default:
        // 'quit', or an answer this build does not know: a close ends the app.
        await performQuit()
    }
  }
  
  /**
   * User chose to minimize to tray
   */
  async function minimizeToTray(): Promise<void> {
    showCloseDialog.value = false
    
    if (rememberChoice.value) {
      await setCloseMode('minimize_to_tray')
    }
    
    // Hide the window
    const window = getCurrentWindow()
    await window.hide()
  }
  
  /**
   * User chose to quit completely
   */
  async function quitCompletely(): Promise<void> {
    showCloseDialog.value = false
    
    if (rememberChoice.value) {
      await setCloseMode('quit_completely')
    }
    
    await performQuit()
  }
  
  /**
   * Quit: perform_quit stops the daemon, then exits.
   */
  async function performQuit(): Promise<void> {
    try {
      await invoke('perform_quit')
    } catch (e) {
      console.error('Failed to perform quit:', e)
      // perform_quit is refused until the app's state is managed. exit()
      // needs no state (process:default grants it), and the exit hook in
      // lib.rs still stops a daemon that was started by then. The window's
      // destroy() is not granted at all, so it did nothing here.
      await exit(0)
    }
  }
  
  /**
   * Cancel the close dialog
   */
  function cancelClose(): void {
    showCloseDialog.value = false
    rememberChoice.value = false
  }
  
  return {
    showCloseDialog,
    closeMode: readonly(closeMode),
    rememberChoice,
    loadCloseMode,
    setCloseMode,
    handleClose,
    minimizeToTray,
    quitCompletely,
    performQuit,
    cancelClose,
  }
}
