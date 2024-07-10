import asyncio
import tensorflow as tf
import numpy as np
from battle import Battle
from teams import OUR_TEAM, DEF_TEAM, NAME_TO_ID_DICT
from gymnasium.spaces import Box, Space
from gymnasium.utils.env_checker import check_env
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from tabulate import tabulate
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
#from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import (
    Gen4EnvSinglePlayer,
    Gen9EnvSinglePlayer,
    MaxBasePowerPlayer,
    ObservationType,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    background_cross_evaluate,
    background_evaluate_player,
)

class vs_cynthia_pokebot(Gen4EnvSinglePlayer):
    def __init__(self, battle_format, opponent, team, start_challenging=False): # Make sure to pass in gen4ou as battle format vs Cynthia
        super().__init__(battle_format=battle_format, opponent=opponent, team=team)
        self.num_battles = 0
        self._ACTION_SPACE = list(range(4 + 5))

    def calc_reward(self, prev_state: Battle, 
                    curr_state: Battle, 
                    starting_value = 0.0, 
                    hp_value: float = 0.20, 
                    fainted_value: float = 0.20, 
                    team_size = 6, 
                    victory_reward = 1.0,
                    slp_reward = 0.15,
                    brn_reward = 0.10,
                    par_reward = 0.10,
                    tox_reward = 0.15) -> float:

        if curr_state not in self._reward_buffer:
            self._reward_buffer[curr_state] = starting_value
        reward = 0

        # Grab our active pokemon
        active_pkmn = curr_state.active_pokemon

        # Want to give a reward if we switch into a pokemon that can do supereffective STAB damage
        good_type_matchup_reward = 0
        our_pkmn_types = (active_pkmn.type_1, active_pkmn.type_2)

        for our_type in our_pkmn_types:
                if our_type:
                    if curr_state.opponent_active_pokemon.damage_multiplier(our_type) == 2:
                        good_type_matchup_reward += 0.5
                    if curr_state.opponent_active_pokemon.damage_multiplier(our_type) == 4:
                        good_type_matchup_reward += 1

        reward += good_type_matchup_reward

        # Want to give a reward if our stats have been raised
        for boost in active_pkmn.boosts:
            reward = reward + active_pkmn.boost[boost] * 0.25

        # Reward for inflicting Sleep/Paralysis/Toxic/Burn
        if curr_state.opponent_active_pokemon.status.SLP: reward += slp_reward
        if curr_state.opponent_active_pokemon.status.BRN: reward += brn_reward
        if curr_state.opponent_active_pokemon.status.PAR: reward += par_reward
        if curr_state.opponent_active_pokemon.status.TOX: reward += tox_reward

        # HP/Fainted rewards for our team
        for pkmn in curr_state.team.values():
            reward += pkmn.current_hp_fraction * hp_value
            if pkmn.fainted:
                reward -= fainted_value

        reward += (team_size - len(curr_state.team)) * hp_value

        # HP/Fainted rewards for defending team
        for pkmn in curr_state.opponent_team.values():
            reward -= pkmn.current_hp_fraction * hp_value
            if pkmn.fainted:
                reward += fainted_value

        reward -= (team_size - len(curr_state.opponent_team)) * hp_value

        # Win/Loss rewards
        if curr_state.won:
            reward += victory_reward
        elif curr_state.lost:
            reward -= victory_reward

        # Value to return
        self._reward_buffer[curr_state] = reward
        return reward - self._reward_buffer[curr_state]

    def embed_battle(self, battle: Battle):
        # -1 indicates that the move does not have a base power or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                # Rescale to a smaller value between 1.0 and 2.0 for training
                move.base_power / 100
            )
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(battle.opponent_active_pokemon.type_1, battle.opponent_active_pokemon.type_2)
                if move.type == battle.active_pokemon.type_1 or move.type == battle.active_pokemon.type_2:
                    moves_dmg_multiplier[i] *= 1.5

        # Counting how many pokemons have not fainted in each team
        n_fainted_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted])
        )
        n_fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted])
        )

        state= np.concatenate([
            [NAME_TO_ID_DICT[str(battle.active_pokemon).split(' ')[0]]],
            [NAME_TO_ID_DICT[str(battle.opponent_active_pokemon).split(' ')[0]]],
            [move_base_power for move_base_power in moves_base_power],
            [move_dmg_multiplier for move_dmg_multiplier in moves_dmg_multiplier],
            [n_fainted_mon_team,
            n_fainted_mon_opponent]])
        
        return state

    def describe_embedding(self) -> Space:
        minimum = [0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        maximum = [5, 5, 2.5, 2.5, 2.5, 2.5, 4, 4, 4, 4, 6, 6]
        return Box(
            np.array(minimum, dtype=np.float32),
            np.array(maximum, dtype=np.float32),
            dtype=np.float32,
        )
    
class drought_typhlosion_pokebot(Gen9EnvSinglePlayer):
    def __init__(self, battle_format, team): # Make sure to pass in gen4ou as battle format vs Cynthia
        super().__init__(battle_format=battle_format, team=team)
        self.num_battles = 0
        self._ACTION_SPACE = list(range(4 + 5))


    def calc_reward(self, prev_state: Battle, curr_state: Battle, starting_value = 0.0, hp_value: float = 0.15, fainted_value: float = 0.15, team_size = 6, victory_reward = 1.0) -> float:
        if curr_state not in self._reward_buffer:
            self._reward_buffer[curr_state] = starting_value
        reward = 0

        # Want to give a reward if we switch into a pokemon that can do supereffective STAB damage
        good_type_matchup_reward = 0
        our_pkmn_types = (curr_state.active_pokemon.type_1, curr_state.active_pokemon.type_2)

        for our_type in our_pkmn_types:
                if our_type:
                    if curr_state.opponent_active_pokemon.damage_multiplier(our_type) == 2:
                        good_type_matchup_reward += 0.5
                    if curr_state.opponent_active_pokemon.damage_multiplier(our_type) == 4:
                        good_type_matchup_reward += 1

        reward += good_type_matchup_reward

        # Want to give a reward if we just did supereffective damage        

        # Want to give a reward if we used a status move which raised stats and still lived

        # Reward for inflicting Sleep/Paralysis/Toxic/Burn

        # HP/Fainted rewards for our team
        for pkmn in curr_state.team.values():
            reward += pkmn.current_hp_fraction * hp_value
            if pkmn.fainted:
                reward -= fainted_value

        reward += (team_size - len(curr_state.team)) * hp_value

        # HP/Fainted rewards for defending team
        for pkmn in curr_state.opponent_team.values():
            reward -= pkmn.current_hp_fraction * hp_value
            if pkmn.fainted:
                reward += fainted_value

        reward -= (team_size - len(curr_state.opponent_team)) * hp_value

        # Win/Loss rewards
        if curr_state.won:
            reward += victory_reward
        elif curr_state.lost:
            reward -= victory_reward


    def embed_battle(self, battle: Battle) -> ObservationType:
        # -1 indicates that the move does not have a base power or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                # Rescale to a smaller value between 1.0 and 2.0 for training
                move.base_power / 100
            )
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(battle.opponent_active_pokemon.type_1, battle.opponent_active_pokemon.type_2)
                if move.type == battle.active_pokemon.type_1 or move.type == battle.active_pokemon.type_2:
                    moves_dmg_multiplier[i] *= 1.5


        # Counting how many pokemons have not fainted in each team
        n_fainted_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted])
        )
        n_fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted])
        )


        state= np.concatenate([
            [NAME_TO_ID_DICT[str(battle.active_pokemon).split(' ')[0]]],
            [NAME_TO_ID_DICT[str(battle.opponent_active_pokemon).split(' ')[0]]],
            [move_base_power for move_base_power in moves_base_power],
            [move_dmg_multiplier for move_dmg_multiplier in moves_dmg_multiplier],
            [n_fainted_mon_team,
            n_fainted_mon_opponent]])
        
        return state

    def describe_embedding(self) -> Space:
        minimum = [0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        maximum = [5, 5, 2.5, 2.5, 2.5, 2.5, 4, 4, 4, 4, 6, 6]
        return Box(
            np.array(minimum, dtype=np.float32),
            np.array(maximum, dtype=np.float32),
            dtype=np.float32,
        )

'''
async def main():
    # First test the environment to ensure the class is consistent
    # with the OpenAI API
    opponent = RandomPlayer(battle_format="gen8randombattle")
    test_env = pokebot(
        battle_format="gen8randombattle", start_challenging=True, opponent=opponent
    )
    check_env(test_env)
    test_env.close()

    # Create one environment for training and one for evaluation
    opponent = RandomPlayer(battle_format="gen8randombattle")
    train_env = pokebot(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )
    opponent = RandomPlayer(battle_format="gen8randombattle")
    eval_env = pokebot(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )

    # Compute dimensions
    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape

    # Create model
    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    # Defining the DQN
    memory = SequentialMemory(limit=10000, window_length=1)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.0,
        nb_steps=10000,
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    # Training the model
    dqn.fit(train_env, nb_steps=10000)
    train_env.close()

    # Evaluating the model
    print("Results against random player:")
    dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    second_opponent = MaxBasePowerPlayer(battle_format="gen8randombattle")
    eval_env.reset_env(restart=True, opponent=second_opponent)
    print("Results against max base power player:")
    dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    eval_env.reset_env(restart=False)

    # Evaluate the player with included util method
    n_challenges = 250
    placement_battles = 40
    eval_task = background_evaluate_player(
        eval_env.agent, n_challenges, placement_battles
    )
    dqn.test(eval_env, nb_episodes=n_challenges, verbose=False, visualize=False)
    print("Evaluation with included method:", eval_task.result())
    eval_env.reset_env(restart=False)

    # Cross evaluate the player with included util method
    n_challenges = 50
    players = [
        eval_env.agent,
        RandomPlayer(battle_format="gen8randombattle"),
        MaxBasePowerPlayer(battle_format="gen8randombattle"),
        SimpleHeuristicsPlayer(battle_format="gen8randombattle"),
    ]
    cross_eval_task = background_cross_evaluate(players, n_challenges)
    dqn.test(
        eval_env,
        nb_episodes=n_challenges * (len(players) - 1),
        verbose=False,
        visualize=False,
    )
    cross_evaluation = cross_eval_task.result()
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    print("Cross evaluation of DQN with baselines:")
    print(tabulate(table))
    eval_env.close()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
'''