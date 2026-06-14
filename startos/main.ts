import { sdk } from './sdk'
import { WEB_UI_PORT } from './interfaces'

export const main = sdk.setupMain(async ({ effects }) => {
  // Mount the persistent volume at /data: config.json, ssh key, datasets, and
  // run outputs (logs + adapters fetched back from the Sparks) all live here.
  const mounts = sdk.Mounts.of().mountVolume({
    volumeId: 'main',
    mountpoint: '/data',
    subpath: null,
    readonly: false,
  })

  const sub = await sdk.SubContainer.of(
    effects,
    { imageId: 'main' },
    mounts,
    'llama-factory-webui',
  )

  return sdk.Daemons.of(effects).addDaemon('webui', {
    subcontainer: sub,
    exec: {
      command: [
        'uvicorn',
        'app:app',
        '--host',
        '0.0.0.0',
        '--port',
        String(WEB_UI_PORT),
      ],
      cwd: '/app',
      env: { LF_DATA_DIR: '/data' },
    },
    ready: {
      display: 'Web Interface',
      fn: () =>
        sdk.healthCheck.checkPortListening(effects, WEB_UI_PORT, {
          successMessage: 'The control panel is ready',
          errorMessage: 'The control panel is not yet listening',
        }),
    },
    requires: [],
  })
})
