import { StartSdk } from '@start9labs/start-sdk'
import { manifest } from './manifest'

/**
 * The bound SDK facade. Import `sdk` everywhere else to reach actions, daemons,
 * interfaces, health checks, file helpers, and the input-form builders.
 */
export const sdk = StartSdk.of().withManifest(manifest).build(true)
