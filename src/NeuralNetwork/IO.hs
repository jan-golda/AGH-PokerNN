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

  readNeuralNetwork :: FilePath -> IO NeuralNetwork
  readNeuralNetwork = fmap (deserializeNetwork . parseFile) . readFile

  writeNeuralNetwork :: FilePath -> NeuralNetwork -> IO ()
  writeNeuralNetwork path network = writeFile path (unparseFile . serializeNetwork $ network)

  randomNeuralNetwork :: Int -> [Int] -> IO NeuralNetwork
  randomNeuralNetwork nInputs [] = return []
  randomNeuralNetwork nInputs (n:rest) = do
      layer   <- randomLayer nInputs n
      network <- randomNeuralNetwork nInputs rest
      return (layer : network)

  randomLayer :: Int -> Int -> IO Layer
  randomLayer nInputs n = do
      weights <- normalMatrix n nInputs
      biases  <- normalMatrix n 1
      return (Layer weights biases)

  ---------------------------------------------------------------------------------
  -- SERIALIZATION FUNCTIONS
  ---------------------------------------------------------------------------------

  deserializeLayer :: [Double] -> [Double] -> Layer
  deserializeLayer weights biases = Layer (toMatrix n weights) (toVector n biases)
      where
        n = length biases

  serializeLayer :: Layer -> [[Double]]
  serializeLayer layer = [Matrix.toList (weights layer), Matrix.toList (biases layer)]

  deserializeNetwork :: [[Double]] -> NeuralNetwork
  deserializeNetwork []         = []
  deserializeNetwork (a:[])     = error "Number of network data rows must be even"
  deserializeNetwork (a:b:rest) = deserializeLayer a b : deserializeNetwork rest

  serializeNetwork :: NeuralNetwork -> [[Double]]
  serializeNetwork []           = []
  serializeNetwork (layer:rest) = serializeLayer layer ++ serializeNetwork rest


  ---------------------------------------------------------------------------------
  -- UTILITY FUNCTIONS
  ---------------------------------------------------------------------------------
  parseLine :: String -> [Double]
  parseLine line = map read (words line)

  unparseLine :: [Double] -> String
  unparseLine list = unwords (map show list)

  parseFile :: String -> [[Double]]
  parseFile file = map parseLine (lines file)

  unparseFile :: [[Double]] -> String
  unparseFile list = unlines (map unparseLine list)

  toVector :: Int -> [a] -> Matrix a
  toVector rows list = Matrix.fromList rows 1 list

  toMatrix :: Int -> [a] -> Matrix a
  toMatrix rows list = Matrix.fromList rows (quot (length list) rows) list

  normalMatrix :: Int -> Int -> IO (Matrix Double)
  normalMatrix x y = fmap (Matrix.fromList x y) Normal.normalsIO