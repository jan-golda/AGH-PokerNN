module PokerNN (loadTrainingSet, stringToTrainingSet, outputToString, stringToInput) where

  import NeuralNetwork.Training

  import Data.Matrix as Matrix
  
  import Data.Char as Char

  ---------------------------------------------------------------------------------
  -- MAIN FUNCTIONS
  ---------------------------------------------------------------------------------

  -- | Creates training set from data in file
  loadTrainingSet :: FilePath -> Int -> IO TrainingSet
  loadTrainingSet path n = fmap (intListToTrainingSet . (take (n*11))) (loadTrainingData path)
  
  -- | Converts raw String TrainingSet representation to actual TrainingSet
  stringToTrainingSet :: String -> Int -> TrainingSet
  stringToTrainingSet string casesNumber = take casesNumber $ intListToTrainingSet ( map (read :: String -> Int) (words string) )
  
  stringToInput :: String -> Matrix Double
  stringToInput inputStr = intListToMatrixInput.
                                    ( foldl (++) [] ).
                                    ( map ( \(r:s:[]) ->
                                    [ fst.head $ ( filter (\x -> (snd x) == s) suits ),
                                      fst.head $ ( filter (\x -> (snd x) == r) ranks ) ]
                                    ) ).words.
                                    (map (Char.toUpper)) $ inputStr
      where suits = [ (1, 'H'), (2, 'S'), (3, 'D'), (4, 'C') ]
            ranks = [ (1, 'A'), (2, '2'), (3, '3'), (4, '4'), (5, '5'), (6, '6'), (7, '7'), (8, '8'), (9, '9'), (10, 'T'), (11, 'J'), (12, 'Q'), (13, 'K') ]

  -- | Converts network output to readable String
  outputToString :: Matrix Double -> String
  outputToString output = foldl (++) "" $ map ( \(x, y) -> x ++ ": " ++ (show . truncate $ ( y * 100 ) ) ++ "%\n" ) $ zip pokerHands (Matrix.toList output)
    where pokerHands = [ "High card",
                         "One pair",
                         "Two pairs",
                         "Three of a kind",
                         "Straight",
                         "Flush",
                         "Full house",
                         "Four of a kind",
                         "Straight flush",
                         "Royal flush" ]
                         
  ---------------------------------------------------------------------------------
  -- UTILITY FUNCTIONS
  ---------------------------------------------------------------------------------

  -- | Loads list of ints from file
  loadTrainingData :: FilePath -> IO [Int]
  loadTrainingData path = fmap (map (read :: String -> Int)) (fmap words (readFile path))

  -- | Converts Int list TestCase input representation to proper Matrix compatible with NeuralNetwork input
  intListToMatrixInput :: [Int] -> Matrix Double
  intListToMatrixInput intListInput =
      let
          suits = map fst ( filter (\x -> even (snd x)) (zip intListInput [0..9]) )
          ranks = map fst ( filter (\x -> odd (snd x)) (zip intListInput [0..9]) )
          cards = [(suit - 1) * 13 + (rank - 1) | (suit, rank) <- zip suits ranks]
          presenceList = map (\x -> if x `elem` cards then 1.0 else 0.0) (take 52 [0..])
      in
          Matrix.fromList 52 1 presenceList

  -- | Converts Int TestCase output respresentation to Matrix representing network's expected output
  intToMatrixOutput :: Int -> Matrix Double
  intToMatrixOutput intOutput =
      let
          handsList = map (\x -> if x == intOutput then 1.0 else 0.0) (take 10 [0..])
      in
          Matrix.fromList 10 1 handsList

  -- | Converts list of Ints (first 11 elements) to single TrainingCase
  intListToTrainingCase :: [Int] -> TrainingCase
  intListToTrainingCase intList = TrainingCase (intListToMatrixInput intListInput) (intToMatrixOutput intOutput)
      where
          intListInput = take 10 intList
          intOutput = head.reverse.(take 11) $ intList

  -- | Converts list of ints to TrainingSet
  intListToTrainingSet :: [Int] -> TrainingSet
  intListToTrainingSet list =
      if length list < 11 then []
                          else [ intListToTrainingCase list ] ++ (intListToTrainingSet (drop 11 list))
