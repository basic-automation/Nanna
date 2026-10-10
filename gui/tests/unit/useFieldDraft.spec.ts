import { describe, expect, it } from 'vitest'
import { nextTick, ref } from 'vue'
import { useFieldDraft } from '~/composables/useFieldDraft'

/**
 * The board reloads on every card event, and a one-way bound field lost the
 * human's half-typed text on each reload. The draft must hold it while the
 * field is focused, and follow the store otherwise.
 */
describe('useFieldDraft', () => {
  function setup() {
    const stored = ref('stored text')
    const card = ref(1)
    const field = useFieldDraft(() => stored.value, () => card.value)
    return { stored, card, field }
  }

  it('keeps typed text through a reload while the field is focused', async () => {
    const { stored, field } = setup()
    field.focus()
    field.draft.value = 'half-typed'
    stored.value = 'stored text, reloaded by an agent post'
    await nextTick()
    expect(field.draft.value).toBe('half-typed')
  })

  it('follows the store while the field is not being edited', async () => {
    const { stored, field } = setup()
    stored.value = 'changed by another member'
    await nextTick()
    expect(field.draft.value).toBe('changed by another member')
  })

  it('an untouched field catches up on blur; an edited one waits for its save', async () => {
    const { stored, field } = setup()
    field.focus()
    stored.value = 'changed meanwhile'
    await nextTick()
    field.blur()
    expect(field.draft.value).toBe('changed meanwhile')

    field.focus()
    field.draft.value = 'my edit'
    field.blur()
    expect(field.draft.value).toBe('my edit')
    stored.value = 'my edit'
    await nextTick()
    expect(field.draft.value).toBe('my edit')
  })

  it('opening another card replaces the draft even mid-edit', async () => {
    const { stored, card, field } = setup()
    field.focus()
    field.draft.value = 'typed on card 1'
    stored.value = 'card 2 text'
    card.value = 2
    await nextTick()
    expect(field.draft.value).toBe('card 2 text')
    stored.value = 'card 2 text, updated'
    await nextTick()
    expect(field.draft.value).toBe('card 2 text, updated')
  })
})
