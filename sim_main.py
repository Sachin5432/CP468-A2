import datetime

from sim.agent import Agent
from sim.world import World

if __name__ == '__main__':
    world = World()

    # world.draw_street_graph()  # uncomment this line to view the graph representing the roads

    # the starting position can be any integer in the closed interval [0, 15].
    agent = Agent(start=0, dest=15, max_walking_dist=3.0)
    use_q_learning = True

    print(f"[{datetime.datetime.now().astimezone()}] Simulation started.")

    while True:
        if not use_q_learning:
            agent.act(world) # Uses dijkstra's algorithm
        else:
            agent.act_q_learning(world) # Uses q-learning

        if agent.finished():
            print(f"[{datetime.datetime.now().astimezone()}] Agent finished.")
            break

        world.update()

    print(f"[{datetime.datetime.now().astimezone()}] Simulation finished.")
