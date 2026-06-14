import { sdk } from './sdk'

export const WEB_UI_PORT = 8080

/**
 * Expose the orchestrator web UI as a StartOS interface (Tor + LAN), so the
 * user can open it from the StartOS dashboard.
 */
export const setInterfaces = sdk.setupInterfaces(async ({ effects }) => {
  const multi = sdk.MultiHost.of(effects, 'web')
  const origin = await multi.bindPort(WEB_UI_PORT, { protocol: 'http' })
  const ui = sdk.createInterface(effects, {
    name: 'Web UI',
    id: 'webui',
    description:
      'The LLaMA-Factory control panel: configure and launch fine-tunes on ' +
      'your DGX Sparks and watch training logs.',
    type: 'ui',
    username: null,
    path: '',
    query: {},
    schemeOverride: null,
    masked: false,
  })
  return [await origin.export([ui])]
})
