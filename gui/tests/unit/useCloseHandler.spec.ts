const { invoke, exit, appWindow } = vi.hoisted(() => ({
  invoke: vi.fn(),
  exit: vi.fn(),
  appWindow: { hide: vi.fn(), destroy: vi.fn() },
}))

vi.mock('@tauri-apps/api/core', () => ({ invoke: (...args: unknown[]) => invoke(...args) }))
vi.mock('@tauri-apps/api/window', () => ({ getCurrentWindow: () => appWindow }))
vi.mock('@tauri-apps/plugin-process', () => ({ exit: (...args: unknown[]) => exit(...args) }))

/** What every command that reads the app's state answers before Rust has managed it. */
const NOT_MANAGED = 'state not managed for field `state` on command `perform_quit`'

/** The composable keeps its dialog state per module (one per process). */
async function freshHandler() {
  vi.resetModules()
  const { useCloseHandler } = await import('~/composables/useCloseHandler')
  return useCloseHandler()
}

const calls = (cmd: string) => invoke.mock.calls.filter(([c]) => c === cmd).length

describe('useCloseHandler', () => {
  beforeEach(() => {
    invoke.mockReset()
    exit.mockReset()
    exit.mockResolvedValue(undefined)
    appWindow.hide.mockReset()
    appWindow.destroy.mockReset()
    vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  it('a remembered close choice survives a restart', async () => {
    localStorage.clear()
    invoke.mockImplementation(async (cmd: string) => (cmd === 'get_close_mode' ? 'ask' : null))
    const first = await freshHandler()
    await first.setCloseMode('minimize_to_tray')

    // A new process starts at "ask"; the stored choice is pushed back.
    const second = await freshHandler()
    expect(await second.loadCloseMode()).toBe('minimize_to_tray')
    expect(invoke).toHaveBeenCalledWith('set_close_mode', { mode: 'minimize_to_tray' })
    localStorage.clear()
  })

  it('quits through perform_quit, which also stops the daemon', async () => {
    invoke.mockResolvedValue(null)
    const handler = await freshHandler()
    await handler.performQuit()
    expect(calls('perform_quit')).toBe(1)
    expect(exit).not.toHaveBeenCalled()
  })

  it('still ends the app when perform_quit cannot run yet', async () => {
    // Before AppState is managed (Config::load can wait on a keyring unlock
    // prompt) perform_quit is refused. destroy() is not in the capability
    // set, so it did nothing; process:default grants exit.
    invoke.mockRejectedValue(NOT_MANAGED)
    const handler = await freshHandler()
    await handler.performQuit()
    expect(exit).toHaveBeenCalledWith(0)
    expect(appWindow.destroy).not.toHaveBeenCalled()
  })

  it('ends the app on a close whose preference cannot be read yet', async () => {
    invoke.mockRejectedValue(NOT_MANAGED)
    const handler = await freshHandler()
    await handler.handleClose()
    expect(exit).toHaveBeenCalledWith(0)
    expect(handler.showCloseDialog.value).toBe(false)
  })

  it('quits once in "quit" mode', async () => {
    invoke.mockImplementation(async (cmd: string) => (cmd === 'handle_window_close' ? 'quit' : null))
    const handler = await freshHandler()
    await handler.handleClose()
    expect(calls('perform_quit')).toBe(1)
  })

  it('asks in "ask" mode, and does nothing more after hiding to the tray', async () => {
    invoke.mockResolvedValueOnce('ask')
    const handler = await freshHandler()
    await handler.handleClose()
    expect(handler.showCloseDialog.value).toBe(true)

    handler.cancelClose()
    invoke.mockResolvedValueOnce('minimized')
    await handler.handleClose()
    expect(handler.showCloseDialog.value).toBe(false)
    expect(calls('perform_quit')).toBe(0)
    expect(exit).not.toHaveBeenCalled()
  })
})
