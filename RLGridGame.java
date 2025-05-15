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

    // Corrected Q-table type
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

    StepResult step(int x, int y, int action) {
        int newX = x, newY = y;
        switch (action) {
            case 0: newY = Math.max(0, y - 1); break; // Up
            case 1: newY = Math.min(GRID_SIZE - 1, y + 1); break; // Down
            case 2: newX = Math.max(0, x - 1); break; // Left
            case 3: newX = Math.min(GRID_SIZE - 1, x + 1); break; // Right
        }

        int reward = -1;
        boolean done = false;

        if (grid[newY][newX] == 1) {
            reward = -100;
            done = true;
        } else if (grid[newY][newX] == 2) {
            reward = 100;
            done = true;
        }

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
