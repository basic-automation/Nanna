import { expect, test } from './fixtures/test-base'

/**
 * Page smoke matrix — each route loads primary content (no white/blank shell).
 * Prefer semantic landmarks; fall back to distinctive copy if headings lag client-side load.
 */
const PAGES: Array<{ path: string; hit: RegExp }> = [
  { path: '/', hit: /Start a new conversation|Welcome|Nanna|type a message/i },
  { path: '/board', hit: /No cards yet|To do/i },
  { path: '/agents', hit: /Agent Overview|Agents|No agents/i },
  { path: '/channels', hit: /Channels|Telegram|Discord|Slack/i },
  { path: '/memory', hit: /Memory|No memories|memories/i },
  { path: '/model-stats', hit: /Model Stats|No model|tokens/i },
  { path: '/scheduler', hit: /Scheduler|No tasks|cron|heartbeat/i },
  { path: '/settings', hit: /Settings|Models|Providers/i },
  { path: '/tool-stats', hit: /Tool Stats|No tool|calls/i },
  { path: '/tools', hit: /Tools|skills|No tools/i },
  { path: '/workspaces', hit: /Workspaces|No workspace|Open Workspace/i },
  { path: '/logs', hit: /Logs|Live|Copy/i },
]

for (const { path, hit } of PAGES) {
  test(`smoke: ${path}`, async ({ page, mock }) => {
    await mock.gotoWithMock(path)
    await expect(page.locator('#__nuxt')).toBeAttached({ timeout: 20_000 })
    // Scoped to <main>: the shell's workspace select carries hidden
    // "Open Workspace…" option text that would otherwise be the first match.
    await expect(page.locator('main').getByText(hit).first()).toBeVisible({ timeout: 25_000 })
    const textLen = await page.locator('body').innerText().then((t) => t.trim().length)
    expect(textLen).toBeGreaterThan(20)
  })
}

/**
 * Same matrix, but reached the way a user reaches it: one page load, then
 * client-side navigation through the rail. Each route is a lazily-imported
 * chunk, so this is the only path that exercises loading a route into an
 * already-running shell — a failure there blanks the whole window instead of
 * rendering a page-level error. Workspaces regressed exactly this way when
 * `@tauri-apps/plugin-dialog` reached Vite's dep optimizer mid-session.
 */
/** The nui rail names every route directly — no overflow flyout. */
const RAIL_LABELS: Record<string, string> = {
  '/board': 'Board',
  '/agents': 'Agents',
  '/channels': 'Channels',
  '/memory': 'Memory',
  '/model-stats': 'Model Stats',
  '/scheduler': 'Scheduler',
  '/settings': 'Settings',
  '/tool-stats': 'Tool Stats',
  '/tools': 'Tools',
  '/workspaces': 'Workspaces',
  '/logs': 'Logs',
}

test('client-side nav: every rail route mounts without blanking the shell', async ({ page, mock }) => {
  await mock.gotoWithMock('/')

  for (const { path, hit } of PAGES) {
    if (path === '/') continue

    await page.getByRole('button', { name: RAIL_LABELS[path], exact: true }).click()

    // The rail itself carries several of these words, so scope the content
    // assertion to <main> and confirm the route actually committed.
    await page.waitForURL((url) => url.pathname === path, { timeout: 25_000 })
    await expect(page.locator('main').getByText(hit).first()).toBeVisible({ timeout: 25_000 })
  }
})
