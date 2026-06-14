import { startSdk } from '@start9labs/start-sdk'
import { sdk } from '../sdk'

/**
 * Runs a one-shot in the orchestrator image that SSHes into the configured
 * Spark(s) and reports `nvidia-smi` + whether the LLaMA-Factory image is built.
 * Reuses orchestrator/spark_client.py so the SSH logic lives in one place.
 */
export const testConnection = sdk.Action.withoutInput(
  'test-connection',

  async ({ effects }) => ({
    name: 'Test Spark Connection',
    description: 'SSH into the configured Spark(s) and verify GPU + image access.',
    warning: null,
    allowedStatuses: 'any',
    group: null,
    visibility: 'enabled',
  }),

  async ({ effects }) => {
    const mounts = sdk.Mounts.of().mountVolume({
      volumeId: 'main',
      mountpoint: '/data',
      subpath: null,
      readonly: true,
    })

    let output: string
    try {
      const { stdout, stderr } = await startSdk.runCommand<typeof sdk.manifest>(
        effects,
        { imageId: 'main' },
        ['python3', '/app/spark_client.py', 'test'],
        { mounts, env: { LF_DATA_DIR: '/data' } },
        'spark-test',
      )
      output =
        (stdout?.toString() || '').trim() +
        (stderr?.toString().trim() ? '\n\n[stderr]\n' + stderr.toString().trim() : '')
    } catch (e: any) {
      output =
        'Connection test failed.\n\n' +
        (e?.stdout?.toString() || '') +
        (e?.stderr?.toString() || e?.message || String(e))
    }

    return {
      version: '1',
      title: 'Spark Connection Test',
      message: 'Result of probing your Spark(s) over SSH.',
      result: { type: 'single', value: output || '(no output)', copyable: true, qr: false, masked: false },
    }
  },
)
