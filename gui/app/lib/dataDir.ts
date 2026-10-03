/**
 * Where the daemon keeps its store (`[general] data_dir`), as the GUI's
 * `get_data_dir` / `set_data_dir` commands report it.
 *
 * The setting is read by the daemon at boot, and no data is moved when it
 * changes — so every wording here says both, rather than letting a user think
 * their memories moved with the folder.
 */
export interface DataDirInfo {
  /** The directory the saved config selects (override or platform default). */
  effective: string
  /** The platform default. */
  default: string
  /** Whether the saved config names somewhere other than the default. */
  is_custom: boolean
  /**
   * The folder the running daemon opened its store in (it reads the setting
   * only at boot); `null` when the daemon could not be asked.
   */
  in_use: string | null
}

/** One-line description of where data lives now. */
export function dataDirSummary(info: DataDirInfo): string {
  return info.is_custom ? 'Custom location' : 'Platform default'
}

/**
 * The notice to show while the saved location differs from the folder the
 * running daemon is using, or `null` while they agree (or the daemon could
 * not say). The daemon reads the setting only at boot, so a difference is a
 * change waiting for a restart — whenever it was made.
 */
export function dataDirRestartNotice(info: DataDirInfo): string | null {
  if (info.in_use === null || info.in_use === info.effective) {
    return null
  }
  return `Nanna keeps using ${info.in_use} until the daemon restarts, then uses ${info.effective}. `
    + 'Existing data is not moved — copy it there first if you want to keep it.'
}

/** The question asked before saving a new location. */
export function dataDirConfirmMessage(current: string, chosen: string): string {
  return `Use ${chosen} for Nanna's data? From the next daemon restart Nanna reads and writes there. `
    + `Your current data stays in ${current} and is not moved.`
}
