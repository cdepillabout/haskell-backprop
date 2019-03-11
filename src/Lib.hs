{-# LANGUAGE DataKinds #-}
{-# LANGUAGE EmptyCase #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

{-# OPTIONS_GHC -Wall #-}

module Lib where

import Data.Kind (Type)
import Data.Singletons.Prelude
import Dependent.Walrus.Matrix
import Dependent.Walrus.Peano
import Dependent.Walrus.Vec


-- data Network hs where
--   OutputLayer
--     :: Vec o Float  -- ^ biases
--     -> Network '[o]
--   HiddenLayer
--     :: Vec i Float -- ^ biases
--     -> Matrix '[h, i] Float -- ^ weights
--     -> Network (h ': hs)  -- ^ the next layer in the network
--     -> Network (i ': h ': hs)
--   InputLayer
--     :: Matrix '[h, i] Float -- ^ weights
--     -> Network (h ': hs) -- ^ the next layer in the network
--     -> Network (i ': h ': hs)



-- data Activations hs where
--   OutputActiviations
--     :: Vec o Float
--     -> Activations '[o]
--   InputActivations
--     :: Vec l Float
--     -> Activations hs
--     -> Activations (l ': hs)

data Network :: [Peano] -> Type where
  End :: Network '[x]
  Layer
    :: Vec o Float -- ^ biases
    -> Matrix '[o, i] Float -- ^ weights
    -> Network (o ': bs) -- ^ next layers
    -> Network (i ': o ': bs)

exampleNetwork :: Network '[N3, N4, N2]
exampleNetwork =
  Layer (replicateVec_ @N4 0) (replicateMatrix_ @'[N4, N3] 0) $
  Layer (replicateVec_ @N2 0) (replicateMatrix_ @'[N2, N4] 0) $
  End

data Activations :: [Peano] -> Type where
  ActivationsStart :: Vec i Float -> Activations '[i]
  ActivationsLayer :: Activations hs -> Vec i Float -> Activations (hs ++ '[i])

data ReverseActivations :: [Peano] -> Type where
  ReverseActivationsEnd
    :: Vec i Float -> ReverseActivations '[i]
  ReverseActivationsLayer
    :: Vec i Float -> ReverseActivations hs -> ReverseActivations (i ': hs)

appendLayerToReverseActivations
  :: ReverseActivations hs -> Vec i Float -> ReverseActivations (hs ++ '[i])
appendLayerToReverseActivations (ReverseActivationsEnd v) newEndV =
  ReverseActivationsLayer v $ ReverseActivationsEnd newEndV
appendLayerToReverseActivations (ReverseActivationsLayer v next) newEndV =
  ReverseActivationsLayer v $ appendLayerToReverseActivations next newEndV

activationsToReverse :: Activations hs -> ReverseActivations hs
activationsToReverse (ActivationsStart v) = ReverseActivationsEnd v
activationsToReverse (ActivationsLayer next v) =
  appendLayerToReverseActivations (activationsToReverse next) v


feedForward
  :: forall i j hs
   . Vec i Float -- ^ x
  -> Network (i ': j ': hs) -- ^ network of weights and biases
  -> ReverseActivations (i ': j ': hs)
feedForward x net =
  ReverseActivationsLayer x $ (go x net :: ReverseActivations (j ': hs))
  where
    go
      :: forall prevActLen something lalas
       . Vec prevActLen Float
      -> Network (prevActLen ': something ': lalas)
      -> ReverseActivations (something ': lalas)
    go prevAct (Layer biases weights End) =
      let z = computeZ weights prevAct biases
          activation = sigmoid z
      in ReverseActivationsEnd activation
    go prevAct (Layer biases weights nextLayers@Layer{}) =
      let z = computeZ weights prevAct biases
          activation = sigmoid z
      in ReverseActivationsLayer activation $ go activation nextLayers


computeZ :: Matrix '[n, m] Float -> Vec m Float -> Vec n Float -> Vec n Float
computeZ weights prevAct biases = zipWithVec (+) (dotProdMatrix weights prevAct) biases

-- | 
--
-- >>> let Just vec = fromListVec_ @N3 [3, 0, -1]
-- >>> sigmoid vec
-- 0.9525... :* (0.5 :* (0.2689... :* EmptyVec))
sigmoid :: Floating a => Vec n a -> Vec n a
sigmoid = fmap ((1 /) . (+ 1) . exp . negate)

--         activation = x
--         activations = [x] # list to store all the activations, layer by layer
--         zs = [] # list to store all the z vectors, layer by layer
--         for b, w in zip(self.biases, self.weights):
--             z = np.dot(w, activation)+b
--             zs.append(z)
--             activation = sigmoid(z)
--             activations.append(activation)
