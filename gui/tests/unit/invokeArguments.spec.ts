import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'

/**
 * Every literal `invoke('name', { ... })` must pass the argument names the Rust command takes.
 *
 * Tauri v2 maps a command's snake_case parameters to camelCase keys and IGNORES keys it does not
 * know. So a misnamed key is no error anywhere: an `Option` parameter silently arrives as `None`,
 * and a required one fails at runtime with a message the call site swallows. That is how the Tools
 * page shipped sending `code` to `create_user_tool` / `update_user_tool`, which take `source`:
 * creating a tool from the GUI never worked, and saving an edit updated only the description while
 * the page marked the code as saved. The Playwright mock answers `true` to any arguments, so no
 * test could see it.
 *
 * Checked: every key a call passes is a parameter of the command, and every non-`Option`
 * parameter is passed. Calls whose argument object is not a plain literal (a variable, a spread)
 * are skipped — they are counted, so the check cannot silently stop seeing calls.
 */

const GUI_ROOT = process.cwd()
const APP_DIR = join(GUI_ROOT, 'app')
const COMMANDS_DIR = join(GUI_ROOT, 'src-tauri/src')

/** Bounds: far above the tree's size (~120 frontend files, ~40 Rust files); cap a runaway walk. */
const DIRECTORY_DEPTH_MAX = 12
const SOURCE_FILES_MAX = 2000

/** Parameters Tauri injects itself; a call never passes them. */
const INJECTED_TYPES = ['State<', 'AppHandle', 'Window', 'WebviewWindow', 'Webview', 'tauri::ipc::Channel']

function collect(directory: string, extensions: string[], depth = 0, found: string[] = []): string[] {
  if (depth > DIRECTORY_DEPTH_MAX) return found
  for (const name of readdirSync(directory)) {
    if (found.length >= SOURCE_FILES_MAX) return found
    const path = join(directory, name)
    if (statSync(path).isDirectory()) collect(path, extensions, depth + 1, found)
    else if (extensions.some((extension) => path.endsWith(extension))) found.push(path)
  }
  return found
}

/** Index of the bracket closing the one at `open`, counting only `open`/`close` characters. */
function matching(source: string, open: number, opener: string, closer: string): number {
  let depth = 0
  for (let index = open; index < source.length; index += 1) {
    if (source[index] === opener) depth += 1
    else if (source[index] === closer) {
      depth -= 1
      if (depth === 0) return index
    }
  }
  return -1
}

/** Split `text` on commas that sit at bracket depth 0. */
function splitTopLevel(text: string): string[] {
  const parts: string[] = []
  let depth = 0
  let start = 0
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index]!
    if ('([{<'.includes(char)) depth += 1
    else if (')]}>'.includes(char)) depth -= 1
    else if (char === ',' && depth === 0) {
      parts.push(text.slice(start, index))
      start = index + 1
    }
  }
  parts.push(text.slice(start))
  return parts.map((part) => part.trim()).filter(Boolean)
}

const camel = (snake: string) => snake.replace(/_([a-z0-9])/g, (_, letter: string) => letter.toUpperCase())

interface CommandSignature { keys: Set<string>, required: Set<string> }

/** Every `#[tauri::command]` fn's camelCase parameter keys and which of them are required. */
function readCommandSignatures(files: string[]): Map<string, CommandSignature> {
  const commands = new Map<string, CommandSignature>()
  for (const file of files) {
    const source = readFileSync(file, 'utf8')
    for (const attribute of source.matchAll(/#\[tauri::command\]/g)) {
      const fnMatch = /\bfn\s+([a-z_][a-z0-9_]*)\s*(?:<[^(]*>)?\s*\(/.exec(source.slice(attribute.index))
      if (!fnMatch) continue
      const open = attribute.index + fnMatch.index + fnMatch[0].length - 1
      const close = matching(source, open, '(', ')')
      if (close < 0) continue
      const keys = new Set<string>()
      const required = new Set<string>()
      for (const parameter of splitTopLevel(source.slice(open + 1, close).replace(/\/\/[^\n]*/g, ''))) {
        const colon = parameter.indexOf(':')
        if (colon < 0) continue
        const name = parameter.slice(0, colon).replace(/^mut\s+/, '').trim()
        const type = parameter.slice(colon + 1).trim()
        if (INJECTED_TYPES.some((injected) => type.includes(injected))) continue
        const key = camel(name.replace(/^_+/, ''))
        keys.add(key)
        if (!type.startsWith('Option<')) required.add(key)
      }
      commands.set(fnMatch[1]!, { keys, required })
    }
  }
  return commands
}

interface InvokeCall { file: string, command: string, keys: string[] | null }

/** Every `invoke('name', …)` call; `keys` is null when its argument is not a plain object literal. */
function readInvokeCalls(files: string[]): InvokeCall[] {
  const calls: InvokeCall[] = []
  for (const file of files) {
    const source = readFileSync(file, 'utf8')
    for (const call of source.matchAll(/\binvoke\s*(?:<[^>]*>)?\s*\(\s*['"]([A-Za-z_]\w*)['"]\s*([,)])/g)) {
      const where = relative(GUI_ROOT, file).replace(/\\/g, '/')
      if (call[2] === ')') {
        calls.push({ file: where, command: call[1]!, keys: [] })
        continue
      }
      const rest = source.slice(call.index + call[0].length).trimStart()
      if (!rest.startsWith('{')) {
        calls.push({ file: where, command: call[1]!, keys: null })
        continue
      }
      const start = source.indexOf('{', call.index + call[0].length)
      const end = matching(source, start, '{', '}')
      const literal = source
        .slice(start + 1, end)
        .replace(/\/\*[\s\S]*?\*\//g, '')
        .replace(/\/\/[^\n]*/g, '')
      const entries = splitTopLevel(literal)
      if (end < 0 || entries.some((entry) => entry.startsWith('...'))) {
        calls.push({ file: where, command: call[1]!, keys: null })
        continue
      }
      const keys = entries.map((entry) => entry.split(':')[0]!.trim().replace(/^['"]|['"]$/g, ''))
      calls.push({ file: where, command: call[1]!, keys })
    }
  }
  return calls
}

const commands = readCommandSignatures(collect(COMMANDS_DIR, ['.rs']))
const calls = readInvokeCalls(collect(APP_DIR, ['.vue', '.ts']))
const checked = calls.filter((call) => call.keys !== null && commands.has(call.command))

describe('tauri invoke arguments', () => {
  // Negative space: an empty read would make the checks below pass vacuously.
  it('reads the command signatures and the calls', () => {
    expect(commands.size).toBeGreaterThan(50)
    expect(commands.get('create_user_tool')?.keys.has('source')).toBe(true)
    expect(commands.get('create_user_tool')?.required.has('source')).toBe(true)
    expect(commands.get('create_user_tool')?.keys.has('state')).toBe(false)
    expect(checked.length).toBeGreaterThan(100)
  })

  it('passes only keys the command takes', () => {
    const unknown = checked.flatMap((call) =>
      call.keys!
        .filter((key) => !commands.get(call.command)!.keys.has(key))
        .map((key) => `${call.file}: invoke('${call.command}', { ${key} })`))
    expect(unknown).toEqual([])
  })

  it('passes every key the command requires', () => {
    const missing = checked.flatMap((call) =>
      [...commands.get(call.command)!.required]
        .filter((key) => !call.keys!.includes(key))
        .map((key) => `${call.file}: invoke('${call.command}') lacks ${key}`))
    expect(missing).toEqual([])
  })
})
