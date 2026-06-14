import { VersionGraph } from '@start9labs/start-sdk'
import { v_0_1_0 } from './v_0_1_0'

/** The current version MUST be the first argument (`current`). */
export const versions = VersionGraph.of({
  current: v_0_1_0,
  other: [],
})
