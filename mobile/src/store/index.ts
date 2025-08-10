/**
 * Redux Store Configuration
 */

import { createStore, combineReducers, applyMiddleware } from 'redux';
import { persistStore, persistReducer } from 'redux-persist';
import AsyncStorage from '@react-native-community/async-storage';
import thunk from 'redux-thunk';

// Reducers
const mentalHealthReducer = (state = {}, action: any) => {
  switch (action.type) {
    case 'UPDATE_MENTAL_HEALTH_STATUS':
      return { ...state, status: action.payload };
    default:
      return state;
  }
};

const userReducer = (state = {}, action: any) => {
  switch (action.type) {
    case 'SET_USER_PROFILE':
      return { ...state, profile: action.payload };
    default:
      return state;
  }
};

const rootReducer = combineReducers({
  mentalHealth: mentalHealthReducer,
  user: userReducer,
});

const persistConfig = {
  key: 'root',
  storage: AsyncStorage,
};

const persistedReducer = persistReducer(persistConfig, rootReducer);

export const store = createStore(persistedReducer, applyMiddleware(thunk));
export const persistor = persistStore(store);

export type RootState = ReturnType<typeof rootReducer>;