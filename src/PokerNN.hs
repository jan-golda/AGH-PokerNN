module PokerNN (stringToTrainingSet, outputToString, stringToInput) where

  import NeuralNetwork.Training

  import Data.Matrix as Matrix

  import Data.Char as Char

  ---------------------------------------------------------------------------------
  -- MAIN FUNCTIONS
  ---------------------------------------------------------------------------------
  
  -- | Converts raw String TrainingSet representation to actual TrainingSet
  stringToTrainingSet :: String -> Int -> TrainingSet
  stringToTrainingSet string casesNumber = take casesNumber $ intListToTrainingSet ( map (read :: String -> Int) (words string) )

  -- | Converts regular poker hand String representation to network input (representation format: "QC, TD, AS, 2H, 3H" -> queen of clubs, ten of diamonds, ace of spades, deuce of hearts, three of hearts)
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

  {-
  -- Int list representation of test case input:
  -- "1s,1r,2s,2r,3s,3r,4s,4r,5s,5r"
  -- where:
  --    ks - suit of k-th card (1 - Hearts, 2 - Spades, 3 - Diamonds, 4 - Clubs)
  --    kr - rank of k-th card (1 - Ace, 2 - 2, 3 - 3, ..., 12 - Queen, 13 - King)
  --
  -- Int representation of test case output is a single digit representing hands as follows:
  -- 0: Nothing in hand (high card); not a recognized poker hand
  -- 1: One pair; one pair of equal ranks within five cards
  -- 2: Two pairs; two pairs of equal ranks within five cards
  -- 3: Three of a kind; three equal ranks within five cards
  -- 4: Straight; five cards, sequentially ranked with no gaps
  -- 5: Flush; five cards with the same suit
  -- 6: Full house; pair + different rank three of a kind
  -- 7: Four of a kind; four equal ranks within five cards
  -- 8: Straight flush; straight + flush
  -- 9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush
  --
  -- Matrix representation of input is a single column matrix with 52 rows, each row representing presence (1) or absence (0) of a card in our 5-card hand
  -- First 13 rows represent Hearts, next 13 - Spades, next - Diamonds and finally Clubs
  --
  -- Matrix representation of output is a singla column matrix with 10 rows, all zeros except for row corresponding with proper hand strength
  -}
