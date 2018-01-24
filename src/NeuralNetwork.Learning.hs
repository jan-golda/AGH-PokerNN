module NeuralNetwork.Learning where

  import NeuralNetwork as NN
  import Data.Matrix as Matrix

  ---------------------------------------------------------------------------------
  -- DEFINITIONS
  ---------------------------------------------------------------------------------
  sigmoid' :: Double -> Double
  --cost :: Matrix Double -> Matrix Double -> Double

  layerError :: Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
  lastLayerError :: Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double

  costDerivativeWithRespectToBiases :: Matrix Double -> Matrix Double
  costDerivativeWithRespectToWeights :: Matrix Double -> Matrix Double -> Matrix Double

  learn :: Matrix Double -> Matrix Double -> Double -> NeuralNetwork -> NeuralNetwork

  weightedInput :: Layer -> Matrix Double -> Matrix Double
  layerOutput :: Layer -> Matrix Double -> Matrix Double

  ---------------------------------------------------------------------------------
  -- IMPLEMENTATION
  ---------------------------------------------------------------------------------
  sigmoid' x = exp(x) / (1+exp(x))^2

  --cost output expected = foldl (+) 0 (Matrix.elementwise (\a b -> (a-b)^2 / 2) output expected)

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
                        error = lastLayerError (weightedInput layer) (layerOutput layer input) expcted


  learn input expected learningRate (layer1:layer2:network) =

  weightedInput layer input = Matrix.elementwise (+) (Matrix.multStd (weights layer) input) (biases layer)

  layerOutput layer input = Matrix.mapCol (\_ x -> sigmoid x) 1 (weightedInput layer input)
