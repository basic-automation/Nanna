import { describe, expect, it } from 'vitest'
import { dataDirConfirmMessage, dataDirRestartNotice, dataDirSummary, type DataDirInfo } from '~/lib/dataDir'

const HOME_DIR = '/home/u/.local/share/nanna'
const platform: DataDirInfo = { effective: HOME_DIR, default: HOME_DIR, is_custom: false, in_use: HOME_DIR }
const custom: DataDirInfo = { effective: '/mnt/big/nanna', default: HOME_DIR, is_custom: true, in_use: HOME_DIR }

describe('data location wording', () => {
  it('names a custom location and the default apart', () => {
    expect(dataDirSummary(platform)).toBe('Platform default')
    expect(dataDirSummary(custom)).toBe('Custom location')
  })

  it('says nothing while the daemon uses the saved location, or cannot be asked', () => {
    expect(dataDirRestartNotice(platform)).toBeNull()
    expect(dataDirRestartNotice({ ...custom, in_use: null })).toBeNull()
  })

  it('says a saved change waits for a restart and moves no data', () => {
    const notice = dataDirRestartNotice(custom)
    expect(notice).toContain(`keeps using ${HOME_DIR} until the daemon restarts`)
    expect(notice).toContain('then uses /mnt/big/nanna')
    expect(notice).toContain('not moved')
    // After the restart the daemon uses it: nothing pending any more.
    expect(dataDirRestartNotice({ ...custom, in_use: custom.effective })).toBeNull()
  })

  it('asks with both folders named', () => {
    const question = dataDirConfirmMessage(platform.effective, custom.effective)
    expect(question).toContain(platform.effective)
    expect(question).toContain(custom.effective)
    expect(question).toContain('not moved')
  })
})
