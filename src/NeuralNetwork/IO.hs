-- Format of file containing NeuralNetwork (extension .nn):
--    Each layer consists of two lines in file:
--        weights - list of wieghts, grouped by neurons
--        biases - list of biases

module NeuralNetwork.IO (readNeuralNetwork, writeNeuralNetwork, deserializeLayer, serializeLayer, deserializeNetwork, serializeNetwork, randomNeuralNetwork, randomLayer) where

  import NeuralNetwork

  import Data.Matrix as Matrix
  import Data.Random.Normal as Normal

  ---------------------------------------------------------------------------------
  -- MAIN FUNCTIONS
  ---------------------------------------------------------------------------------

  -- | Reads NeuralNetwork from given file
  readNeuralNetwork :: FilePath -> IO NeuralNetwork
  readNeuralNetwork = fmap (deserializeNetwork . parseFile) . readFile

  -- | Writes NeuralNetwork to given file
  writeNeuralNetwork :: FilePath -> NeuralNetwork -> IO ()
  writeNeuralNetwork path network = writeFile path (unparseFile . serializeNetwork $ network)

  -- | Generates random NeuralNetwork using normal distribution
  randomNeuralNetwork :: Int -> [Int] -> IO NeuralNetwork
  randomNeuralNetwork nInputs [] = return []
  randomNeuralNetwork nInputs (n:rest) = do
      layer   <- randomLayer nInputs n
      network <- randomNeuralNetwork nInputs rest
      return (layer : network)

  -- | Generates random Layer using normal distribution
  randomLayer :: Int -> Int -> IO Layer
  randomLayer nInputs n = do
      weights <- normalDistributionMatrix n nInputs
      biases  <- normalDistributionMatrix n 1
      return (Layer weights biases)

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
  parseLine :: String -> [Double]
  parseLine line = map read (words line)

  -- | Parses list of doubles to string
  unparseLine :: [Double] -> String
  unparseLine list = unwords (map show list)

  -- | Parses string to list of lists of doubles, where each sub list represents line in given string
  parseFile :: String -> [[Double]]
  parseFile file = map parseLine (lines file)

  -- | Parses list of lists of doubles to string, where each line represents sub list
  unparseFile :: [[Double]] -> String
  unparseFile list = unlines (map unparseLine list)

  -- | Creates one dimentional matrix with given size filled wih data from list
  toVector :: Int -> [a] -> Matrix a
  toVector rows list = Matrix.fromList rows 1 list

  -- | Creates matrix with given number of rows filled with data from list, number of columns is calculated as size of list divided by  number of rows
  toMatrix :: Int -> [a] -> Matrix a
  toMatrix rows list = Matrix.fromList rows (quot (length list) rows) list

  -- | Creates matrix of given size filled with random normal distribution numbers
  normalDistributionMatrix :: Int -> Int -> IO (Matrix Double)
  normalDistributionMatrix x y = fmap (Matrix.fromList x y) Normal.normalsIO
