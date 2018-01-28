-- Format of NeuralNetwork String representation:
--    Each layer consists of two lines in file:
--        weights - list of wieghts, grouped by neurons
--        biases - list of biases

module NeuralNetwork (NeuralNetwork, NetworkCostGradient, Layer(Layer, weights, biases), feed, stringToNetwork, networkToString, calculateCostGradient) where

  import Data.Matrix as Matrix

  ---------------------------------------------------------------------------------
  -- TYPE DEFINITIONS
  ---------------------------------------------------------------------------------

  data Layer = Layer {weights :: Matrix Double, biases :: Matrix Double} deriving (Show)

  type NeuralNetwork = [Layer]
  
  -- | Type that has exactly the same structure as NeuralNetwork, but is intended to hold cost function gradient for whole network
  type NetworkCostGradient = NeuralNetwork

  ---------------------------------------------------------------------------------
  -- MAIN FUNCTIONS
  ---------------------------------------------------------------------------------

  -- | Calculates output of neural network by feeding signal forward through each layer
  feed :: NeuralNetwork -> Matrix Double -> Matrix Double
  feed []     input = input
  feed (l:ls) input = feed ls (layerOutput l input)

  {-
  -- | Performs network learning on a given dataset with gradient descent cost function minimization
  learn :: Matrix Double -> Matrix Double -> Double -> NeuralNetwork -> NeuralNetwork
  learn input expected learningRate []      = []
  learn input expected learningRate network = newNetwork
      where (newNetwork, _) = backpropagation input expected learningRate network
      -}
        
  -- | Converts String representation of network to actual NeuralNetwork
  stringToNetwork :: String -> NeuralNetwork
  stringToNetwork = deserializeNetwork . parseFile

  -- | Converts NeuralNetwork to String representation
  networkToString :: NeuralNetwork -> String
  networkToString = unparseFile . serializeNetwork

  ---------------------------------------------------------------------------------
  -- NETWORK FUNCTIONS
  ---------------------------------------------------------------------------------

  -- | Calculates output vector for a single layer
  layerOutput :: Layer -> Matrix Double -> Matrix Double
  layerOutput layer input = Matrix.mapCol (\_ x -> sigmoid x) 1 (weightedInput layer input)

  -- | Calculates weighted input for a single layer (sum of all weights * inputs + bias)
  weightedInput :: Layer -> Matrix Double -> Matrix Double
  weightedInput layer input = Matrix.elementwise (+) (Matrix.multStd (weights layer) input) (biases layer)

  -- | Sigmoid function used for calculating single neuron's output
  sigmoid :: Double -> Double
  sigmoid x = 1/(1 + exp (-x))

  -- | Cost function used for estimating network performance
  cost :: Matrix Double -> Matrix Double -> Double
  cost output expected = 1/(2*n) * (foldl (+) 0 $ Matrix.toList (Matrix.elementwise costF output expected))
      where
        costF = \x y -> (x - y) ** 2.0
        n = fromIntegral(Matrix.nrows output)

  -- | Sigmoid function derivative
  sigmoid' :: Double -> Double
  sigmoid' x = exp(x) / (1+exp(x))^2


  -- | Calculates error vector for a single layer (should not be applied to the last layer of a network)
  layerError :: Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
  layerError weightedInput nextWeights nextError =
      Matrix.elementwise
        (\a b -> a * sigmoid'(b))
        (multStd (Matrix.transpose nextWeights) nextError)
        weightedInput

  -- | Calculates error vector for last layer in network
  lastLayerError :: Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
  lastLayerError weightedInput output expected =
      Matrix.elementwise
        (\a b -> a * sigmoid'(b))
        (Matrix.elementwise (\a b -> b-a) output expected)
        weightedInput


  -- | Calculates vector of cost function partial derivatives with respect to biases for a single layer
  costDerivativeWithRespectToBiases :: Matrix Double -> Matrix Double
  costDerivativeWithRespectToBiases error = error

  -- | Calculates vector of cost function partial derivatives with respect to weights for a single layer
  costDerivativeWithRespectToWeights :: Matrix Double -> Matrix Double -> Matrix Double
  costDerivativeWithRespectToWeights prevOutput error = Matrix.fromList height width values
        where
          height = Matrix.nrows error
          width = Matrix.nrows prevOutput
          values = [a * b | a <- errorList, b <- prevOutputList]
              where
                errorList = Matrix.toList error
                prevOutputList = Matrix.toList prevOutput

{-
  -- | Applies backpropagation learning algorithm to neural network
  backpropagation :: Matrix Double -> Matrix Double -> Double -> NeuralNetwork -> (NeuralNetwork, Matrix Double)
  
  backpropagation _ _ _ [] = error "Cannot perform backpropagation on empty network"
  
  backpropagation input expected learningRate (layer:[]) =
        let
            error = lastLayerError (weightedInput layer input) (layerOutput layer input) expected
            derWeights = costDerivativeWithRespectToWeights input error
            derBiases = costDerivativeWithRespectToBiases error
            newWeights = Matrix.elementwise (+) (weights layer) (Matrix.scaleMatrix learningRate derWeights)
            newBiases = Matrix.elementwise (+) (biases layer) (Matrix.scaleMatrix learningRate derBiases)
        in
            ([ Layer newWeights newBiases ], error)
            
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
            -}
            
  ---------------------------------------------------------------------------------
  -- BACKPROPAGATION RELATED FUNCTIONS
  ---------------------------------------------------------------------------------

  -- | Applies backpropagation learning algorithm to neural network
  costBackpropagation :: NeuralNetwork -> Matrix Double -> Matrix Double -> (NetworkCostGradient, Matrix Double)
  
  --costBackpropagation _ _ _ [] = error "Cannot perform backpropagation on empty network"
  
  costBackpropagation (layer:[]) input expected =
        let
            error = lastLayerError (weightedInput layer input) (layerOutput layer input) expected
            derWeights = costDerivativeWithRespectToWeights input error
            derBiases = costDerivativeWithRespectToBiases error
        in
            ([ Layer derWeights derBiases ], error)
            
  costBackpropagation (layer1:layer2:rest) input expected =
        let
            (gradient, nextError) = costBackpropagation (layer2:rest) (layerOutput layer1 input) expected
            error = layerError (weightedInput layer1 input) (weights layer2) nextError
            derWeights = costDerivativeWithRespectToWeights input error
            derBiases = costDerivativeWithRespectToBiases error
        in
            ([ Layer derWeights derBiases ] ++ gradient, error)
  
  calculateCostGradient :: NeuralNetwork -> Matrix Double -> Matrix Double -> NetworkCostGradient
  calculateCostGradient network input expected = gradient
    where (gradient, _) = costBackpropagation network input expected
  
  ---------------------------------------------------------------------------------
  -- SERIALIZATION FUNCTIONS
  ---------------------------------------------------------------------------------

  -- | Creates Layer from two lists of doubles, first containing wegihts, second containing biases
  deserializeLayer :: [Double] -> [Double] -> Layer
  deserializeLayer weights biases = Layer (toMatrix n weights) (toVector n biases)
      where
        n = length biases

  -- | Creates list of doubles from Layer, in format: [[--weights--],[--biases--]]
  serializeLayer :: Layer -> [[Double]]
  serializeLayer layer = [Matrix.toList (weights layer), Matrix.toList (biases layer)]

  -- | Creates NeuralNetwork from list of lists of doubles, number of lists has to be even
  deserializeNetwork :: [[Double]] -> NeuralNetwork
  deserializeNetwork []         = []
  deserializeNetwork (a:[])     = error "Number of network data rows must be even"
  deserializeNetwork (a:b:rest) = deserializeLayer a b : deserializeNetwork rest

  -- | Creates list of lists of doubles from NeuralNetwork, returned list will always have even number of lists
  serializeNetwork :: NeuralNetwork -> [[Double]]
  serializeNetwork []           = []
  serializeNetwork (layer:rest) = serializeLayer layer ++ serializeNetwork rest

  ---------------------------------------------------------------------------------
  -- UTILITY FUNCTIONS
  ---------------------------------------------------------------------------------

  -- | Parses string to list of doubles
  stringToDoubleList :: String -> [Double]
  stringToDoubleList line = map read (words line)

  -- | Parses list of doubles to string
  doubleListToString :: [Double] -> String
  doubleListToString list = unwords (map show list)

  -- | Parses string to list of lists of doubles, where each sub list represents line in given string
  parseFile :: String -> [[Double]]
  parseFile file = map stringToDoubleList (lines file)

  -- | Parses list of lists of doubles to string, where each line represents sub list
  unparseFile :: [[Double]] -> String
  unparseFile list = unlines (map doubleListToString list)

  -- | Creates one dimentional matrix with given size filled wih data from list
  toVector :: Int -> [a] -> Matrix a
  toVector rows list = Matrix.fromList rows 1 list

  -- | Creates matrix with given number of rows filled with data from list, number of columns is calculated as size of list divided by  number of rows
  toMatrix :: Int -> [a] -> Matrix a
  toMatrix rows list = Matrix.fromList rows (div (length list) rows) list
  

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
