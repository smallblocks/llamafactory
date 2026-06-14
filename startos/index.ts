import { sdk } from './sdk'
import { versions } from './versions'
import { actions } from './actions'
import { setInterfaces } from './interfaces'

// Required ABI exports for a StartOS service package.
export { manifest } from './manifest'
export { main } from './main'
export { actions } from './actions'

// Back up the whole volume (config, ssh key, datasets, fetched adapters).
// setupBackups yields the createBackup export plus a restoreInit InitScript
// that must be composed into init so restores are applied on startup.
export const { createBackup, restoreInit } = sdk.setupBackups(async () =>
  sdk.Backups.ofVolumes('main'),
)

// init composes: version migrations, action registration, interface export,
// and backup restore.
export const init = sdk.setupInit(versions, actions, setInterfaces, restoreInit)
export const uninit = sdk.setupUninit(versions)
