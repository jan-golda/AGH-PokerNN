module NeuralNetwork.Training (TrainingCase(TrainingCase, input, expected), TrainingSet, trainOnCase, trainOnSet, trainingSetShuffle, epochTraining) where

  import NeuralNetwork

  import System.Random.Shuffle as Shuffle
  import Data.Matrix as Matrix

  ---------------------------------------------------------------------------------
  -- TYPE DEFINITIONS
  ---------------------------------------------------------------------------------

  data TrainingCase = TrainingCase {input :: Matrix Double, expected :: Matrix Double} deriving (Show)

  type TrainingSet = [TrainingCase]

  ---------------------------------------------------------------------------------
  -- MAIN FUNCTIONS
  ---------------------------------------------------------------------------------

  -- | Trains network on a single training case
  trainOnCase :: NeuralNetwork -> TrainingCase -> Double -> NeuralNetwork
  trainOnCase network trainingCase learningRate = trainOnSet network [trainingCase] learningRate

  {-
  -- | Trains network on given dataset
  trainOnSet :: NeuralNetwork -> TrainingSet -> Double -> NeuralNetwork
  trainOnSet network [] _ = network
  trainOnSet network (x:xs) learningRate = trainOnSet (trainOnCase network x learningRate) xs learningRate
  -}

  -- | Trains network on given TrainingSet
  trainOnSet :: NeuralNetwork -> TrainingSet -> Double -> NeuralNetwork
  trainOnSet network trainingSet learningRate =
      let
         totalGradient = totalCostGradient network trainingSet
         n = fromIntegral (length trainingSet) :: Double
         descent = scaleGradient totalGradient ((-learningRate)/(2*n))
      in applyGradientDescent network descent

  -- | Performs repetitive training over given TrainingSet
  epochTraining :: NeuralNetwork -> TrainingSet -> Double -> Int -> NeuralNetwork
  epochTraining network _ _ 0 = network
  epochTraining network trainingSet learningRate epochsNumber = epochTraining newNetwork trainingSet learningRate (epochsNumber - 1)
    where newNetwork = trainOnSet network trainingSet learningRate

  ---------------------------------------------------------------------------------
  -- UTILITY FUNCTIONS
  ---------------------------------------------------------------------------------

  -- | Sums NetworkCostGradients from all TrainingCases in TrainingSet
  totalCostGradient :: NeuralNetwork -> TrainingSet -> NetworkCostGradient
  totalCostGradient _ [] = error "Cannot calculate NetworkCostGradient for empty TrainingSet"
  totalCostGradient network (lastCase : []) = calculateCostGradient network (input lastCase) (expected lastCase)
  totalCostGradient network (testCase : set) = sumCostGradient (calculateCostGradient network (input testCase) (expected testCase)) (totalCostGradient network set)

  -- | Sums two NetworkCostGradients
  sumCostGradient :: NetworkCostGradient -> NetworkCostGradient -> NetworkCostGradient
  sumCostGradient (lastA : []) (lastB : []) = [ Layer (Matrix.elementwise (+) (weights lastA) (weights lastB)) (Matrix.elementwise (+) (biases lastA) (biases lastB)) ]
  sumCostGradient (layerA : restA) (layerB : restB) = [ Layer (Matrix.elementwise (+) (weights layerA) (weights layerB)) (Matrix.elementwise (+) (biases layerA) (biases layerB)) ] ++ (sumCostGradient restA restB)

  -- | Adds properly scaled gradient to all network's weights and biases
  applyGradientDescent :: NeuralNetwork -> NetworkCostGradient -> NeuralNetwork
  applyGradientDescent network gradient = sumCostGradient network gradient

  -- | Multiplies NetworkCostGradient by given scalar
  scaleGradient :: NetworkCostGradient -> Double -> NetworkCostGradient
  scaleGradient [] _ = []
  scaleGradient (layer : rest) scalar = [(Layer scaledWeights scaledBiases)] ++ (scaleGradient rest scalar)
    where
        scaledWeights = Matrix.fromList (Matrix.nrows (weights layer)) (Matrix.ncols (weights layer)) $ map (\x -> x * scalar) (Matrix.toList (weights layer))
        scaledBiases = Matrix.fromList (Matrix.nrows (biases layer)) (Matrix.ncols (biases layer)) $ map (\x -> x * scalar) (Matrix.toList (biases layer))

  -- | Shuffles training set
  trainingSetShuffle :: TrainingSet -> IO TrainingSet
  trainingSetShuffle = Shuffle.shuffleM
