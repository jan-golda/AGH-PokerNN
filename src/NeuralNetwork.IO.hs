-- Format of file containing NeuralNetwork (extension .nn):
--    Each layer consists of two lines in file:
--        weights - list of wieghts, grouped by neurons
--        biases - list of biases

module NeuralNetwork.IO (readNeuralNetwork, writeNeuralNetwork, deserializeLayer, serializeLayer, deserializeNetwork, serializeNetwork) where

  import NeuralNetwork
  import Data.Matrix as Matrix

  ---------------------------------------------------------------------------------
  -- MAIN FUNCTIONS
  ---------------------------------------------------------------------------------

  readNeuralNetwork :: FilePath -> IO NeuralNetwork
  readNeuralNetwork = fmap (deserializeNetwork . parseFile) . readFile

  writeNeuralNetwork :: FilePath -> NeuralNetwork -> IO ()
  writeNeuralNetwork path network = writeFile path (unparseFile . serializeNetwork $ network)

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
  
  -- Converts Int list TestCase input representation to proper Matrix compatible with NeuralNetwork input
  
  intListToMatrixInput :: [Int] -> Matrix Double
  intListToMatrixInput intListInput = 
      let
          suits = map fst ( filter (\x -> even (snd x)) (zip intListInput [0..9]) )
          ranks = map fst ( filter (\x -> odd (snd x)) (zip intListInput [0..9]) )
          cards = [(suit - 1) * 13 + (rank - 1) | (suit, rank) <- zip suits ranks]
          presenceList = map (\x -> if x `elem` cards then 1.0 else 0.0) (take 52 [0..])
      in
          Matrix.fromList 52 1 presenceList
              
        
  
  -- Converts Int TestCase output respresentation to Matrix representing network's expected output
  
  intToMatrixOutput :: Int -> Matrix Double
  intToMatrixOutput intOutput =
      let 
          handsList = map (\x -> if x == intOutput then 1.0 else 0.0) (take 10 [0..])
      in
          Matrix.fromList 10 1 handsList
          
  
  -- Converts list of Ints (first 11 elements) to single TrainingCase
  
  intListToTrainingCase :: [Int] -> TrainingCase
  intListToTrainingCase intList = TrainingCase (intListToMatrixInput intListInput) (intToMatrixOutput intOutput)
      where
          intListInput = take 10 intList
          intOutput = head.reverse.(take 11) $ intList
          
          
  intListToTrainingSet :: [Int] -> TrainingSet
  intListToTrainingSet list = 
      if length list < 11 then []
                          else [ intListToTrainingCase list ] ++ (intListToTrainingSet (drop 11 list))
