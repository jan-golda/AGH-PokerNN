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
  
  -- Calculates output vector for a single layer
  layerOutput layer input = Matrix.mapCol (\_ x -> sigmoid x) 1 (weightedInput layer input)

  -- Calculates output of neural network by feeding signal forward through each layer
  feed []     input = input
  feed (l:ls) input = feed ls (layerOutput l input)

  -- Calculates weighted input for a single layer (sum of all weights * input + bias)
  weightedInput layer input = Matrix.elementwise (+) (Matrix.multStd (weights layer) input) (biases layer)

  -- Sigmoid function used for calculating single neuron's output
  sigmoid x = 1/(1 + exp (-x))
  sigmoid' x = exp(x) / (1+exp(x))^2

  -- Calculates error vector for a single layer (should not be applied to the last layer of a network)
  layerError weightedInput nextWeights nextError =
      Matrix.elementwise
        (\a b -> a * sigmoid'(b))
        (multStd (Matrix.transpose nextWeights) nextError)
        weightedInput

  -- Calculates error vector for last layer in network
  lastLayerError weightedInput output expected =
      Matrix.elementwise
        (\a b -> a * sigmoid'(b))
        (Matrix.elementwise (\a b -> b-a) output expected)
        weightedInput

  -- Calculates vector of Cost function partial derivatives with respect to biases for a single layer
  costDerivativeWithRespectToBiases error = error
  
  -- Calculates vector of Cost function partial derivatives with respect to weights for a single layer
  costDerivativeWithRespectToWeights prevOutput error = Matrix.fromList height width values
        where
          height = Matrix.nrows error
          width = Matrix.nrows prevOutput
          values = [a * b | a <- errorList, b <- prevOutputList]
              where
                errorList = Matrix.toList error
                prevOutputList = Matrix.toList prevOutput

                
  -- Applies backpropagation learning algorithm to neural network
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
