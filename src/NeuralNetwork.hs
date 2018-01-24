module NeuralNetwork (NeuralNetwork, Layer, feed, learn) where

  import Data.Matrix as Matrix

  ---------------------------------------------------------------------------------
  -- TYPE DEFINITIONS
  ---------------------------------------------------------------------------------
  data Layer = Layer {weights :: Matrix Double, biases :: Matrix Double} deriving (Show)

  type NeuralNetwork = [Layer]

  ---------------------------------------------------------------------------------
  -- DEFINITIONS
  ---------------------------------------------------------------------------------
  layerOutput :: Layer -> Matrix Double -> Matrix Double

  weightedInput :: Layer -> Matrix Double -> Matrix Double

  sigmoid :: Double -> Double
  sigmoid' :: Double -> Double

  layerError :: Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
  lastLayerError :: Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double

  costDerivativeWithRespectToBiases :: Matrix Double -> Matrix Double
  costDerivativeWithRespectToWeights :: Matrix Double -> Matrix Double -> Matrix Double

  feed :: NeuralNetwork -> Matrix Double -> Matrix Double
  learn :: Matrix Double -> Matrix Double -> Double -> NeuralNetwork -> NeuralNetwork

  ---------------------------------------------------------------------------------
  -- IMPLEMENTATION
  ---------------------------------------------------------------------------------
  layerOutput layer input = Matrix.mapCol (\_ x -> sigmoid x) 1 (weightedInput layer input)

  feed []     input = input
  feed (l:ls) input = networkOutput ls (layerOutput l input)

  weightedInput layer input = Matrix.elementwise (+) (Matrix.multStd (weights layer) input) (biases layer)

  sigmoid x = 1/(1 + exp (-x))
  sigmoid' x = exp(x) / (1+exp(x))^2

  layerError weightedInput nextWeights nextError =
      Matrix.elementwise
        (\a b -> a * sigmoid'(b))
        (multStd (Matrix.transpose nextWeights) nextError)
        weightedInput

  lastLayerError weightedInput output expected =
      Matrix.elementwise
        (\a b -> a * sigmoid'(b))
        (Matrix.elementwise (\a b -> b-a) output expected)
        weightedInput

  costDerivativeWithRespectToBiases error = error

  costDerivativeWithRespectToWeights prevOutput error = Matrix.fromList height width values
        where
          height = Matrix.nrows error
          width = Matrix.nrows prevOutput
          values = [a * b | a <- errorList, b <- prevOutputList]
              where
                errorList = Matrix.toList error
                prevOutputList = Matrix.toList prevOutput

  learn input expected learningRate []                      = []

  learn input expected learningRate (layer:[])              = [ Layer newWeights newBiases ]
        where
          newWeights = Matrix.elementwise (+) (weights layer) (Matrix.scaleMatrix learningRate derWeights)
          newBiases = Matrix.elementwise (+) (biases layer) (Matrix.scaleMatrix learningRate derBiases)
              where
                  derWeights = costDerivativeWithRespectToWeights input error
                  derBiases = costDerivativeWithRespectToBiases error
                      where
                        error = lastLayerError (weightedInput layer) (layerOutput layer input) expected


  learn input expected learningRate (layer1:layer2:network) = network
