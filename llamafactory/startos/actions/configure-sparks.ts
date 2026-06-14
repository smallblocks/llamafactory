import { sdk } from '../sdk'
import { configFile } from '../file-models/config'
import { sshKeyFile, hfTokenFile } from '../file-models/secrets'

const { InputSpec, Value } = sdk

const inputSpec = InputSpec.of({
  primarySparkHost: Value.text({
    name: 'Primary Spark Host',
    description: 'Hostname or IP of the first DGX Spark (reachable over SSH).',
    required: true,
    default: null,
    placeholder: 'spark-01.local',
  }),
  primarySparkUser: Value.text({
    name: 'SSH User',
    description: 'The login user on the Spark (DGX OS default is "nvidia").',
    required: true,
    default: 'nvidia',
  }),
  sshPort: Value.number({
    name: 'SSH Port',
    description: 'SSH port on the Spark.',
    required: true,
    default: 22,
    integer: true,
    min: 1,
    max: 65535,
  }),
  sshPrivateKey: Value.textarea({
    name: 'SSH Private Key',
    description:
      'A private key (PEM/OpenSSH) whose public half is in the Spark user\'s ' +
      '~/.ssh/authorized_keys. Stored in this service\'s private volume and ' +
      'used only to reach your Sparks. Paste the FULL key including header/footer.',
    warning:
      'This is a credential. It is written to the service volume and never ' +
      'shown again. Use a dedicated key for this service.',
    required: true,
    default: null,
    minRows: 6,
    maxRows: 14,
    placeholder: '-----BEGIN OPENSSH PRIVATE KEY-----\n...\n-----END OPENSSH PRIVATE KEY-----',
  }),
  useBothSparks: Value.toggle({
    name: 'Use Both Sparks (2-node)',
    description:
      'Run distributed training across two Sparks over ConnectX/200GbE. ' +
      'Only needed for full fine-tuning or very large models — a 31B LoRA ' +
      'fits comfortably on a single Spark.',
    default: false,
  }),
  secondarySparkHost: Value.text({
    name: 'Secondary Spark Host',
    description: 'Hostname/IP of the second Spark. Required only if "Use Both Sparks" is on.',
    required: false,
    default: null,
    placeholder: 'spark-02.local',
  }),
  dockerImage: Value.text({
    name: 'Spark Image Tag',
    description: 'The LLaMA-Factory image built by spark/build.sh on each Spark.',
    required: true,
    default: 'llamafactory-spark:latest',
  }),
  remoteWorkDir: Value.text({
    name: 'Remote Work Directory',
    description: 'Absolute path on the Spark where run dirs and outputs are written.',
    required: true,
    default: '/home/nvidia/llamafactory-runs',
  }),
  hfToken: Value.text({
    name: 'Hugging Face Token (optional)',
    description:
      'Only needed for gated/private models. Gemma 4 is Apache-2.0, so this ' +
      'is usually blank. Leave empty to keep the existing token unchanged.',
    required: false,
    default: null,
    masked: true,
  }),
})

export const configureSparks = sdk.Action.withInput(
  'configure-sparks',

  async ({ effects }) => ({
    name: 'Configure Sparks',
    description: 'Set the DGX Spark connection details and SSH credentials.',
    warning: null,
    allowedStatuses: 'any',
    group: null,
    visibility: 'enabled',
  }),

  inputSpec,

  // Prefill non-secret fields from existing config. Never prefill the key/token.
  async ({ effects }) => {
    const cfg = await configFile.read().const(effects)
    if (!cfg) return {}
    return {
      primarySparkHost: cfg.primarySparkHost || undefined,
      primarySparkUser: cfg.primarySparkUser,
      sshPort: cfg.sshPort,
      useBothSparks: cfg.useBothSparks,
      secondarySparkHost: cfg.secondarySparkHost ?? undefined,
      dockerImage: cfg.dockerImage,
      remoteWorkDir: cfg.remoteWorkDir,
    }
  },

  async ({ effects, input }) => {
    // Persist the private key to its own file (700/600 enforced in-container).
    await sshKeyFile.write(effects, input.sshPrivateKey.trim() + '\n')

    let hfTokenSet = (await configFile.read().const(effects))?.hfTokenSet ?? false
    if (input.hfToken && input.hfToken.trim()) {
      await hfTokenFile.write(effects, input.hfToken.trim())
      hfTokenSet = true
    }

    await configFile.merge(effects, {
      primarySparkHost: input.primarySparkHost,
      primarySparkUser: input.primarySparkUser,
      sshPort: input.sshPort,
      useBothSparks: input.useBothSparks,
      secondarySparkHost: input.secondarySparkHost,
      dockerImage: input.dockerImage,
      remoteWorkDir: input.remoteWorkDir,
      hfTokenSet,
    })

    return {
      version: '1',
      title: 'Sparks Configured',
      message:
        'Saved. Use "Test Spark Connection" to verify SSH + GPU access, then ' +
        'open the Web UI to launch a fine-tune.',
      result: { type: 'single', value: input.primarySparkHost, copyable: false, qr: false, masked: false },
    }
  },
)
