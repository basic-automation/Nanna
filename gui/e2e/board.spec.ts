import { expect, test } from './fixtures/test-base'

/**
 * The board client (P25 Stage 4) against the hermetic mock: the quick-add
 * line makes a card, a refused line says why and makes nothing, the card
 * view posts on the thread, and Mark done moves the card to Done.
 */
test('quick-add makes a card, the thread takes a post, done moves it', async ({ page, mock }) => {
  await mock.gotoWithMock('/board')
  const quickAdd = page.getByTestId('board-quick-add')
  await expect(quickAdd).toBeVisible({ timeout: 25_000 })

  await quickAdd.fill('Write the changelog #release p2 @me')
  await quickAdd.press('Enter')
  const todo = page.getByTestId('board-column-todo')
  await expect(todo.getByText('Write the changelog')).toBeVisible()
  await expect(todo.getByText('#release')).toBeVisible()
  await expect(todo.getByText('p2')).toBeVisible()

  await quickAdd.fill('Paint the shed @nobody')
  await quickAdd.press('Enter')
  await expect(page.getByTestId('board-quick-add-error')).toContainText('no member @nobody')
  await expect(todo.getByText('Paint the shed')).toHaveCount(0)

  await todo.getByText('Write the changelog').click()
  const view = page.getByTestId('board-card-view')
  await expect(view).toBeVisible()
  await page.getByTestId('card-post').fill('Draft is in CHANGELOG.md')
  await view.getByRole('button', { name: 'Post' }).click()
  await expect(page.getByTestId('card-thread')).toContainText('Draft is in CHANGELOG.md')

  await page.getByTestId('card-sub-card-add').fill('Collect the PR titles p1')
  await page.getByTestId('card-sub-card-add').press('Enter')
  await expect(page.getByTestId('card-sub-cards')).toContainText('Collect the PR titles')
  await expect(page.getByTestId('card-sub-cards')).toContainText('1/1 open')
  // Sub-cards sit inside their parent by default: the board shows the count.
  await expect(todo.getByText('↳ 1/1')).toBeVisible()

  await page.getByTestId('card-done').click()
  await expect(page.getByTestId('board-column-done').getByText('Write the changelog')).toBeVisible()
})

test('Inbox lists cards assigned to me; Members adds an agent', async ({ page, mock }) => {
  await mock.gotoWithMock('/board')
  const quickAdd = page.getByTestId('board-quick-add')
  await expect(quickAdd).toBeVisible({ timeout: 25_000 })
  await quickAdd.fill('Review the PR @me')
  await quickAdd.press('Enter')
  await expect(page.getByTestId('board-column-todo').getByText('Review the PR')).toBeVisible()

  await page.getByTestId('board-view-inbox').click()
  await expect(page.getByTestId('board-inbox')).toContainText('Review the PR')

  await page.getByTestId('board-view-members').click()
  await page.getByTestId('member-new-name').fill('Builder')
  await page.getByTestId('member-new-models').fill('qwen3.5:9b, qwen3.5:9b, claude-sonnet-5')
  await page.getByTestId('member-add').click()
  const builder = page.getByTestId('member-agent:builder')
  await expect(builder).toContainText('qwen3.5:9b, claude-sonnet-5')
})
