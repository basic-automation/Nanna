import { ref, watch, type Ref } from 'vue'

/**
 * A text field's draft of a stored value, kept while the human is typing.
 *
 * A field bound one way (`:value="stored"`, saved on `@change`) loses what was
 * typed whenever its component re-renders: Vue re-applies a `value` prop on
 * every patch and overwrites the live text when it differs from the stored
 * one. The board reloads on every card event — an agent's progress post, a
 * chat run's task write — so a description being typed snapped back to the
 * stored text, and because the field then equalled its starting value its
 * `change` never fired: the edit was lost without a word.
 *
 * The draft follows the stored value while the field is not being edited,
 * and always when `key` changes (another card opened). While it is focused
 * the human's text wins. On blur an untouched draft catches up with whatever
 * was stored meanwhile; an edited one is left for its `change` save, whose
 * stored value then reaches it (re-seeding at once would flash the old text).
 */
export function useFieldDraft(stored: () => string, key: () => unknown): {
  draft: Ref<string>
  focus: () => void
  blur: () => void
} {
  const draft = ref(stored())
  const editing = ref(false)
  let atFocus = draft.value

  watch(key, () => {
    editing.value = false
    draft.value = stored()
  })
  watch(stored, (next) => {
    if (!editing.value) draft.value = next
  })

  return {
    draft,
    focus: () => {
      editing.value = true
      atFocus = draft.value
    },
    blur: () => {
      editing.value = false
      if (draft.value === atFocus) draft.value = stored()
    },
  }
}
