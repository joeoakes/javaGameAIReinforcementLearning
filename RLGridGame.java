/*
This is the AI's learned knowledge. Learned Q-values — the agent's memory
For each state (grid cell), it stores Q-values for four possible actions (up, down, left, right).
It updates the estimated value of an action based on the reward received and the best future action.
This is where the AI learns from trial and error.
chooseAction() Policy function — how the agent chooses what to do
run() Learning loop — how the agent improves over time
This AI learns how to navigate a grid world to reach a goal while avoiding traps, by trial-and-error.
It's not hardcoded like rule-based systems or FSMs — it discovers behavior using rewards.
 */
import javax.swing.*;
import java.awt.*;
import java.util.Random;

// Custom return type for step results
class StepResult {
    int newX, newY, reward;
    boolean done;

    StepResult(int x, int y, int reward, boolean done) {
        this.newX = x;
        this.newY = y;
        this.reward = reward;
        this.done = done;
    }
}

public class RLGridGame extends JPanel implements Runnable {

    final int GRID_SIZE = 5;
    final int TILE_SIZE = 100;
    final int WIDTH = GRID_SIZE * TILE_SIZE;
    final int HEIGHT = GRID_SIZE * TILE_SIZE;

    int[][] grid = new int[GRID_SIZE][GRID_SIZE]; // 0 = empty, 1 = trap, 2 = goal
    int agentX = 0, agentY = 0;

    // Q-table type
    double[][] Q = new double[GRID_SIZE * GRID_SIZE][4]; // Q[state][action]
    double alpha = 0.1;   // learning rate
    double gamma = 0.9;   // discount factor
    double epsilon = 0.2; // exploration rate

    Random rand = new Random();

    public RLGridGame() {
        setPreferredSize(new Dimension(WIDTH, HEIGHT));
        setBackground(Color.WHITE);
        setupGrid();
        new Thread(this).start();
    }

    void setupGrid() {
        grid[4][4] = 2; // Goal
        grid[1][2] = 1; // Trap
        grid[3][3] = 1; // Trap
    }

    int getState(int x, int y) {
        return y * GRID_SIZE + x;
    }

    /*
    x, y: Current position of the agent.
    action: The direction the agent wants to move (0–3).
    0 = Up, 1 = Down, 2 = Left, 3 = Right
     */
    StepResult step(int x, int y, int action) {
        int newX = x, newY = y;
        switch (action) {
            //Math.max() and Math.min() are used to prevent the agent from moving outside the grid boundaries
            case 0: newY = Math.max(0, y - 1); break; // Up
            case 1: newY = Math.min(GRID_SIZE - 1, y + 1); break; // Down
            case 2: newX = Math.max(0, x - 1); break; // Left
            case 3: newX = Math.min(GRID_SIZE - 1, x + 1); break; // Right
        }

        int reward = -1;  //Default reward for a move is -1 (small penalty to encourage shorter paths).
        boolean done = false;  //done is false unless a goal or trap is reached.

        if (grid[newY][newX] == 1) {
            reward = -100;  // trap tile (1), it gets a -100 penalty and the episode ends.
            done = true;
        } else if (grid[newY][newX] == 2) {
            reward = 100;  //goal tile (2), it gets a +100 reward and the episode ends.
            done = true;
        }
        //New agent position, Reward received, Whether the episode is finished
        return new StepResult(newX, newY, reward, done);
    }

    int chooseAction(int x, int y) {
        if (rand.nextDouble() < epsilon) {
            return rand.nextInt(4); // Explore
        }

        int state = getState(x, y);
        double[] qValues = Q[state];
        int bestAction = 0;
        for (int a = 1; a < 4; a++) {
            if (qValues[a] > qValues[bestAction]) {
                bestAction = a;
            }
        }
        return bestAction;
    }

    public void run() {
        int episodes = 1000;

        // Training phase
        for (int e = 0; e < episodes; e++) {
            int x = 0, y = 0;
            boolean done = false;

            while (!done) {
                int action = chooseAction(x, y);
                StepResult result = step(x, y, action);

                int state = getState(x, y);
                int newState = getState(result.newX, result.newY);

                double maxQ = Q[newState][0];
                for (int i = 1; i < 4; i++) {
                    if (Q[newState][i] > maxQ) maxQ = Q[newState][i];
                }

                Q[state][action] += alpha * (result.reward + gamma * maxQ - Q[state][action]);

                x = result.newX;
                y = result.newY;
                done = result.done;
            }
        }

        // Live execution of learned policy
        agentX = 0;
        agentY = 0;
        new Timer(500, e -> {
            if (grid[agentY][agentX] == 2 || grid[agentY][agentX] == 1) return;

            int action = chooseAction(agentX, agentY);
            StepResult result = step(agentX, agentY, action);
            agentX = result.newX;
            agentY = result.newY;

            repaint();
        }).start();
    }

    public void paintComponent(Graphics g) {
        super.paintComponent(g);

        // Draw grid tiles
        for (int y = 0; y < GRID_SIZE; y++) {
            for (int x = 0; x < GRID_SIZE; x++) {
                int type = grid[y][x];
                if (type == 1) g.setColor(Color.RED);         // Trap
                else if (type == 2) g.setColor(Color.GREEN);  // Goal
                else g.setColor(Color.LIGHT_GRAY);            // Empty

                g.fillRect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE);
                g.setColor(Color.BLACK);
                g.drawRect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE);

                // Draw Q-values (skip traps and goal for clarity)
                if (type == 0) {
                    int state = getState(x, y);
                    g.setFont(new Font("Monospaced", Font.PLAIN, 10));

                    // Find best action
                    int bestAction = 0;
                    for (int a = 1; a < 4; a++) {
                        if (Q[state][a] > Q[state][bestAction]) bestAction = a;
                    }

                    // Draw each action
                    for (int action = 0; action < 4; action++) {
                        if (action == bestAction) {
                            g.setColor(Color.BLUE); // Best action
                        } else {
                            g.setColor(Color.DARK_GRAY); // Other actions
                        }

                        String label = "";
                        int q = (int) Q[state][action];
                        int tx = x * TILE_SIZE, ty = y * TILE_SIZE;

                        switch (action) {
                            case 0: label = "↑" + q; g.drawString(label, tx + 40, ty + 15); break;
                            case 1: label = "↓" + q; g.drawString(label, tx + 40, ty + 95); break;
                            case 2: label = "←" + q; g.drawString(label, tx + 5, ty + 55); break;
                            case 3: label = "→" + q; g.drawString(label, tx + 75, ty + 55); break;
                        }
                    }
                }
            }
        }

        // Draw agent
        g.setColor(Color.BLUE);
        g.fillOval(agentX * TILE_SIZE + 30, agentY * TILE_SIZE + 30, 40, 40);
    }


    public static void main(String[] args) {
        JFrame frame = new JFrame("RL Grid World");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new RLGridGame());
        frame.pack();
        frame.setVisible(true);
    }
}
