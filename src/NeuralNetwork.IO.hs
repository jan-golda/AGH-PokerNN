-- Format of file containing NeuralNetwork (extension .nn):
--    Each layer consists of two lines in file:
--        weights - list of wieghts, grouped by neurons
--        biases - list of biases

module NeuralNetwork.IO (readNeuralNetwork) where

  import NeuralNetwork
  import Data.Matrix as Matrix

  toVector :: Int -> [a] -> Matrix a
  toVector rows list = Matrix.fromList rows 1 list

  toMatrix :: Int -> [a] -> Matrix a
  toMatrix rows list = Matrix.fromList rows (quot (length list) rows) list

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

  parseLine :: String -> [Double]
  parseLine line = map read (words line)

  unparseLine :: [Double] -> String
  unparseLine list = unwords (map show list)

  parseFile :: String -> [[Double]]
  parseFile file = map parseLine (lines file)

  unparseFile :: [[Double]] -> String
  unparseFile list = unlines (map unparseLine list)

  readNeuralNetwork :: FilePath -> IO NeuralNetwork
  readNeuralNetwork = fmap (deserializeNetwork . parseFile) . readFile

  writeNeuralNetwork :: FilePath -> NeuralNetwork -> IO ()
  writeNeuralNetwork path network = writeFile path (unparseFile . serializeNetwork $ network)
