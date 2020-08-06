package sus;

import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;

import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;

public class MyBayesNet {
	private int numberOfNodes;

	private BayesNet bayesNet;
	private int[] bayesNetAdjacencyMatrix;
	List<String> names;

	private Instances data;

	MyBayesNet(int nodeNumber, List<String> names, Instances data) throws Exception {
		this.numberOfNodes = nodeNumber;
		this.names = names;
		this.data = data;
	}

	public void generateBayesNetAdjacencyMatrix() throws Exception {
		do {
			bayesNetAdjacencyMatrix = new int[numberOfNodes * numberOfNodes];
			Random seed = new Random();
			int minNumberOfArcs = numberOfNodes - 1;
			int maxNumberOfArcs = numberOfNodes * (numberOfNodes - 1) / 2;
			int arcNumber = seed.nextInt(maxNumberOfArcs - minNumberOfArcs) + minNumberOfArcs;

			for (int i = 0; i < arcNumber; i++) {
				int arcPlacement = seed.nextInt(numberOfNodes * numberOfNodes);
				bayesNetAdjacencyMatrix[arcPlacement] = 1;
			}
		} while (isCyclic());
	}

	public boolean isCyclic() {
		boolean[] visited = new boolean[numberOfNodes];
		boolean[] recStack = new boolean[numberOfNodes];

		for (int i = 0; i < numberOfNodes; i++)
			if (isCyclicUtil(i, visited, recStack))
				return true;

		return false;
	}

	private boolean isCyclicUtil(int i, boolean[] visited, boolean[] recStack) {
		if (recStack[i])
			return true;

		if (visited[i])
			return false;

		visited[i] = true;
		recStack[i] = true;

		int[] children = new int[numberOfNodes];
		int numberOfChildren = 0;
		for (int j = 0; j < numberOfNodes; j++) {
			if (bayesNetAdjacencyMatrix[j * numberOfNodes + i] == 1) {
				children[numberOfChildren] = j;
				numberOfChildren++;
			}

		}
		for (int j = 0; j < numberOfChildren; j++) {
			if (isCyclicUtil(children[j], visited, recStack))
				return true;
		}

		recStack[i] = false;
		return false;
	}
	
	public void createBiffFileFromAdjacencyMatrix() throws Exception {
		String data = "";
		for (int i = 0; i < numberOfNodes; i++) {
			data += ("<VARIABLE TYPE=\"nature\">\n");
			data += ("<NAME>" + names.get(i) + "</NAME>\n");
			data += ("<OUTCOME>Value1</OUTCOME>\n");
			data += ("<OUTCOME>Value2</OUTCOME>\n");
			data += ("<PROPERTY>position = (" + i * 50 + ",0)</PROPERTY>\n");
			data += ("</VARIABLE>\n");
		}

		for (int j = 0; j < numberOfNodes; j++) {
			int numberOfParents = 0;
			data += ("<DEFINITION>\n");
			data += ("<FOR>" + names.get(j) + "</FOR>\n");
			for (int i = 0; i < numberOfNodes; i++) {
				if (bayesNetAdjacencyMatrix[j * numberOfNodes + i] == 1) {
					data += ("<GIVEN>" + names.get(i) + "</GIVEN>\n");
					numberOfParents++;
				}
			}
			data += ("<TABLE>\n");
			int x = (int) Math.pow(2, numberOfParents);
			for (int i = 0; i < x; i++) {
				data += ("0.5 0.5\n");
			}
			data += ("</TABLE>\n");
			data += ("</DEFINITION>\n");
		}
		String biffFile = readFileAsString(".\\data\\scheme.xml");
		biffFile = biffFile.replace("---data---", data);
		try (PrintWriter out = new PrintWriter(".\\data\\result.xml")) {
			out.println(biffFile);
		}
	}
	
	public void generateBayesNet() throws Exception {
		bayesNet = new BayesNet();
		String[] netOptions2 = { "-D", "-Q", "weka.classifiers.bayes.net.search.fixed.FromFile", "--", "-B",
				".\\data\\result.xml", "-E", "weka.classifiers.bayes.net.estimate.SimpleEstimator",
				"--", "-A", "0.5" };
		bayesNet.setOptions(netOptions2);
		bayesNet.buildClassifier(data);
	}

	private static String readFileAsString(String fileName) throws Exception {
		String data = "";
		data = new String(Files.readAllBytes(Paths.get(fileName)));
		return data;
	}

	public void printBayesNetAdjacencyMatrix() {
		for (int i = 0; i < numberOfNodes; i++) {
			for (int j = 0; j < numberOfNodes; j++) {
				System.out.print(bayesNetAdjacencyMatrix[j * numberOfNodes + i] + " ");
			}
			System.out.println();
		}
		System.out.println();
	}
	
	public int[] getBayesNetAdjacencyMatrix() {
		return bayesNetAdjacencyMatrix;
	}

	public void setBayesNetAdjacencyMatrix(int[] bayesNetArray) {
		this.bayesNetAdjacencyMatrix = bayesNetArray;
	}

	public BayesNet getBayesNet() {
		return bayesNet;
	}

	public void setBayesNet(BayesNet bayesNet) {
		this.bayesNet = bayesNet;
	}

}
