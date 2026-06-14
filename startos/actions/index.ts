import { sdk } from '../sdk'
import { configureSparks } from './configure-sparks'
import { trainingDefaults } from './training-defaults'
import { testConnection } from './test-connection'

export const actions = sdk.Actions.of()
  .addAction(configureSparks)
  .addAction(trainingDefaults)
  .addAction(testConnection)
