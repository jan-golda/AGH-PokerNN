module PokerNN where

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
