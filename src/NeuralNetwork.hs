module NeuralNetwork (NeuralNetwork, Layer, feed, learn) where

  import Data.Matrix as Matrix

  ---------------------------------------------------------------------------------
  -- TYPE DEFINITIONS
  ---------------------------------------------------------------------------------
  data Layer = Layer {weights :: Matrix Double, biases :: Matrix Double} deriving (Show)

  type NeuralNetwork = [Layer]


  ---------------------------------------------------------------------------------
  -- FUNCTIONS
  ---------------------------------------------------------------------------------
  
  -- Calculates output vector for a single layer
  
  layerOutput :: Layer -> Matrix Double -> Matrix Double
  layerOutput layer input = Matrix.mapCol (\_ x -> sigmoid x) 1 (weightedInput layer input)
  

  
  -- Calculates output of neural network by feeding signal forward through each layer
  
  feed :: NeuralNetwork -> Matrix Double -> Matrix Double
  feed []     input = input
  feed (l:ls) input = feed ls (layerOutput l input)
  

  
  -- Calculates weighted input for a single layer (sum of all weights * inputs + bias)
  
  weightedInput :: Layer -> Matrix Double -> Matrix Double
  weightedInput layer input = Matrix.elementwise (+) (Matrix.multStd (weights layer) input) (biases layer)
  

  
  -- Sigmoid function used for calculating single neuron's output
  
  sigmoid :: Double -> Double
  sigmoid x = 1/(1 + exp (-x))
  
  
  
  -- Sigmoid function derivative
  
  sigmoid' :: Double -> Double
  sigmoid' x = exp(x) / (1+exp(x))^2
  
  

  -- Calculates error vector for a single layer (should not be applied to the last layer of a network)
  
  layerError :: Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
  layerError weightedInput nextWeights nextError =
      Matrix.elementwise
        (\a b -> a * sigmoid'(b))
        (multStd (Matrix.transpose nextWeights) nextError)
        weightedInput
        
        

  -- Calculates error vector for last layer in network
  
  lastLayerError :: Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
  lastLayerError weightedInput output expected =
      Matrix.elementwise
        (\a b -> a * sigmoid'(b))
        (Matrix.elementwise (\a b -> b-a) output expected)
        weightedInput
        
        

  -- Calculates vector of cost function partial derivatives with respect to biases for a single layer
  
  costDerivativeWithRespectToBiases :: Matrix Double -> Matrix Double
  costDerivativeWithRespectToBiases error = error
  
  
  
  -- Calculates vector of cost function partial derivatives with respect to weights for a single layer
  
  costDerivativeWithRespectToWeights :: Matrix Double -> Matrix Double -> Matrix Double
  costDerivativeWithRespectToWeights prevOutput error = Matrix.fromList height width values
        where
          height = Matrix.nrows error
          width = Matrix.nrows prevOutput
          values = [a * b | a <- errorList, b <- prevOutputList]
              where
                errorList = Matrix.toList error
                prevOutputList = Matrix.toList prevOutput

                
                
  -- Performs network learning on a given dataset with gradient descent cost function minimization 
  
  learn :: Matrix Double -> Matrix Double -> Double -> NeuralNetwork -> NeuralNetwork
  learn input expected learningRate [] = []

  learn input expected learningRate network = newNetwork
    where (newNetwork, _) = backpropagation input expected learningRate network 
  
  
  
  -- Applies backpropagation learning algorithm to neural network
  
  backpropagation :: Matrix Double -> Matrix Double -> Double -> NeuralNetwork -> (NeuralNetwork, Matrix Double)
  backpropagation input expected learningRate (layer1:layer2:rest) = 
        let
            (network, nextError) = backpropagation (layerOutput layer1 input) expected learningRate (layer2:rest)
            error = layerError (weightedInput layer1 input) (weights layer2) nextError
            derWeights = costDerivativeWithRespectToWeights input error
            derBiases = costDerivativeWithRespectToBiases error
            newWeights = Matrix.elementwise (+) (weights layer1) (Matrix.scaleMatrix learningRate derWeights)
            newBiases = Matrix.elementwise (+) (biases layer1) (Matrix.scaleMatrix learningRate derBiases)
        in
            ([ Layer newWeights newBiases ] ++ network, error)
            
                
  backpropagation input expected learningRate (layer:[]) = 
        let
            error = lastLayerError (weightedInput layer input) (layerOutput layer input) expected
            derWeights = costDerivativeWithRespectToWeights input error
            derBiases = costDerivativeWithRespectToBiases error
            newWeights = Matrix.elementwise (+) (weights layer) (Matrix.scaleMatrix learningRate derWeights)
            newBiases = Matrix.elementwise (+) (biases layer) (Matrix.scaleMatrix learningRate derBiases)
        in
            ([ Layer newWeights newBiases ], error)

            
{-
  -- Little network sample for testing purposes
  ll = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
  w1 = Matrix.fromList 3 2 ll
  w2 = Matrix.fromList 2 3 ll
  b1 = Matrix.fromList 3 1 [0.5, 0.5, 0.5]
  b2 = Matrix.fromList 2 1 [0.5, 0.5]
  layer1 = Layer w1 b1
  layer2 = Layer w2 b2
  network = [layer1, layer2]
  input = Matrix.fromList 2 1 [0.0, 1]
  expected = Matrix.fromList 2 1 [1.0, 0]
  ratio = 0.4
  
  test = learn input expected ratio
  -}