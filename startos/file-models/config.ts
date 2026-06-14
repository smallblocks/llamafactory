import { FileHelper, z } from '@start9labs/start-sdk'

/**
 * Orchestrator configuration, persisted to the `main` volume as config.json.
 *
 * Written by the StartOS actions (Configure Sparks / Training Defaults) and
 * read by the Python orchestrator inside the container, which mounts the same
 * volume at /data and reads /data/config.json. Keep field names in sync with
 * orchestrator/app.py.
 *
 * The SSH PRIVATE KEY is intentionally NOT stored here — it lives in its own
 * file (see sshKey.ts) so it is never echoed back in plaintext config dumps.
 */
export const configShape = z.object({
  // --- Spark connection ---
  primarySparkHost: z.string().default(''),
  primarySparkUser: z.string().default('nvidia'),
  sshPort: z.number().int().positive().default(22),
  // Second Spark for 2-node FSDP over ConnectX/200GbE. null = single node.
  secondarySparkHost: z.string().nullable().default(null),
  useBothSparks: z.boolean().default(false),

  // --- Remote execution ---
  // Image built by spark/build.sh on each Spark.
  dockerImage: z.string().default('llamafactory-spark:latest'),
  // Where run dirs (config, dataset, logs, adapters) live on the Spark.
  remoteWorkDir: z.string().default('/home/nvidia/llamafactory-runs'),

  // --- Model / method ---
  model: z.string().default('google/gemma-4-31B-it'),
  // LLaMA-Factory chat template. Use the newest gemma template your
  // LLaMA-Factory build ships (gemma4 if present, else gemma).
  template: z.string().default('gemma'),
  finetuningType: z.enum(['lora', 'full']).default('lora'),

  // --- LoRA / hyperparameters ---
  loraRank: z.number().int().positive().default(16),
  loraAlpha: z.number().int().positive().default(32),
  learningRate: z.string().default('1.0e-4'),
  epochs: z.number().positive().default(3),
  cutoffLen: z.number().int().positive().default(4096),
  perDeviceBatchSize: z.number().int().positive().default(1),
  gradAccumSteps: z.number().int().positive().default(8),
  // 4 -> QLoRA (bitsandbytes 4-bit). null -> bf16 LoRA (recommended on a Spark;
  // 128GB unified memory fits a 31B LoRA without quantization).
  quantizationBit: z.number().int().nullable().default(null),

  // --- Auth flags (the token itself lives in hfToken.ts) ---
  hfTokenSet: z.boolean().default(false),
})

export type Config = z.infer<typeof configShape>

export const configFile = FileHelper.json('./config.json', configShape)
