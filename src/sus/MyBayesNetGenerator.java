package sus;

import java.awt.BorderLayout;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import javax.swing.JFrame;

import weka.classifiers.bayes.BayesNet;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.gui.graphvisualizer.BIFFormatException;
import weka.gui.graphvisualizer.GraphVisualizer;

public class MyBayesNetGenerator {
	public static int pathNumber = 1;
	public static String[] path = {".\\data\\vote",
			".\\data\\diabetes",
			".\\data\\breast-cancer",
			".\\data\\hypothyroid",
			".\\data\\segment-challenge"};
	private int numberOfNodes;

	private List<String> names;
	private Instances data;
	private Instances dataEval;

	private int numberOfIndividuals = 20;
	private List<MyBayesNet> population;
	private List<MyBayesNet> newPopulation;

	MyBayesNetGenerator() throws BIFFormatException, Exception {
		for (int j = 0; j < 4; j++) {
			pathNumber = j;
			data = loadData(path[pathNumber] + ".arff");
			dataEval = loadData(path[pathNumber] + "_eval.arff");
			
			setBayesNetStructureData();

			initializePopulation();
			for (int i = 0; i < 20; i++) {
				//System.out.println("Generation " + i + ": ");
				evolution();
			}

			drawGraph(population.get(0).getBayesNet(), "Best " + j);
			System.out.println("Dataset " + j);
			System.out.println("my correct percentage: " + checkOnEvalData(population.get(0).getBayesNet(), dataEval));
			//System.out.println("my bayes score: " + population.get(0).getBayesNet().measureBayesScore());
			//System.out.println("my aic score: " + population.get(0).getBayesNet().measureAICScore());
			//System.out.println("my mdl score: " + population.get(0).getBayesNet().measureMDLScore());
			System.out.println();
			
			createWekaBayesNet();
		}
		
		

	}

	private void initializePopulation() throws Exception {
		population = new ArrayList<MyBayesNet>();
		newPopulation = new ArrayList<MyBayesNet>();
		for (int i = 0; i < numberOfIndividuals; i++) {
			MyBayesNet newBayesNet = new MyBayesNet(numberOfNodes, names, data);
			newBayesNet.generateBayesNetAdjacencyMatrix();
			newBayesNet.createBiffFileFromAdjacencyMatrix();
			newBayesNet.generateBayesNet();
			// newBayesNet.printBayesNetAdjacencyMatrix();
			// drawGraph(newBayesNet.getBayesNet(),"i");
			population.add(newBayesNet);
		}
	}

	private void evolution() throws Exception {
		population.sort(new SortByFitness());
		/*for (Iterator<MyBayesNet> i = population.iterator(); i.hasNext();) {
			MyBayesNet item = i.next();
			System.out.println("Fitness: " + getFitness(item.getBayesNet(), dataEval));
		}*/

		int eliteNumber = 5;
		int numberOfMutations = 5;
		for (int i = 0; i < eliteNumber; i++) {
			newPopulation.add(population.get(i));
		}
		for (int i = eliteNumber; i < numberOfIndividuals - numberOfMutations; i += 2) {
			crossover(population.get(i), population.get(i + 1));
		}
		for(int i = numberOfIndividuals - numberOfMutations; i < numberOfIndividuals; i++) {
			mutation(population.get(i));
		}

	}

	private void crossover(MyBayesNet parent, MyBayesNet parent2) throws Exception {
		int sizeOfAdjacencyMatrix = numberOfNodes * numberOfNodes;
		int[] parentAdjacencyMatrix = parent.getBayesNetAdjacencyMatrix();
		int[] parent2AdjacencyMatrix = parent2.getBayesNetAdjacencyMatrix();

		int[] childAdjacencyMatrix = new int[sizeOfAdjacencyMatrix];
		int[] child2AdjacencyMatrix = new int[sizeOfAdjacencyMatrix];

		MyBayesNet child = new MyBayesNet(numberOfNodes, names, data);
		MyBayesNet child2 = new MyBayesNet(numberOfNodes, names, data);

		Random seed = new Random();
		do {
			int crossoverPlacement = seed.nextInt(sizeOfAdjacencyMatrix);
			for (int i = 0; i < sizeOfAdjacencyMatrix; i++) {
				if (i < crossoverPlacement) {
					childAdjacencyMatrix[i] = parentAdjacencyMatrix[i];
					child2AdjacencyMatrix[i] = parent2AdjacencyMatrix[i];
				} else {
					childAdjacencyMatrix[i] = parent2AdjacencyMatrix[i];
					child2AdjacencyMatrix[i] = parentAdjacencyMatrix[i];
				}
			}
			child.setBayesNetAdjacencyMatrix(childAdjacencyMatrix);
			child2.setBayesNetAdjacencyMatrix(child2AdjacencyMatrix);

			/*
			 * System.out.println("child 1"); child.printBayesNetAdjacencyMatrix();
			 * System.out.println("child 2"); child2.printBayesNetAdjacencyMatrix();
			 */

		} while (child.isCyclic() && child2.isCyclic());
		child.createBiffFileFromAdjacencyMatrix();
		child.generateBayesNet();
		newPopulation.add(child);
		child2.createBiffFileFromAdjacencyMatrix();
		child2.generateBayesNet();
		newPopulation.add(child2);
	}
	
	private void mutation(MyBayesNet beforeMutation) throws Exception {
		int sizeOfAdjacencyMatrix = numberOfNodes * numberOfNodes;
		int[] beforeAdjacencyMatrix = beforeMutation.getBayesNetAdjacencyMatrix();
		int[] afterAdjacencyMatrix = new int[sizeOfAdjacencyMatrix];
		
		
		MyBayesNet afterMutation = new MyBayesNet(numberOfNodes, names, data);
		Random seed = new Random();
		do {
			for(int i = 0; i < sizeOfAdjacencyMatrix; i++) {
				afterAdjacencyMatrix[i] = beforeAdjacencyMatrix[i];
			}
			int mutationPlacement = seed.nextInt(sizeOfAdjacencyMatrix);
			if(afterAdjacencyMatrix[mutationPlacement] == 1) {
				afterAdjacencyMatrix[mutationPlacement] = 0;
			} else {
				afterAdjacencyMatrix[mutationPlacement] = 1;
			}
			afterMutation.setBayesNetAdjacencyMatrix(afterAdjacencyMatrix);
			
			/*
			 * System.out.println("before xxx");
			 * beforeMutation.printBayesNetAdjacencyMatrix();
			 * System.out.println("after mutationPlacement: " + mutationPlacement);
			 * afterMutation.printBayesNetAdjacencyMatrix();
			 */
		}while(afterMutation.isCyclic());
		afterMutation.createBiffFileFromAdjacencyMatrix();
		afterMutation.generateBayesNet();
		newPopulation.add(afterMutation);
	}
	
	static double checkOnEvalData(BayesNet bayesNet, Instances dataEval) throws Exception {
		double numberOfInstances = dataEval.numInstances();
		double numberOfCorrectClassification = 0;
		for (int i = 0; i < dataEval.numInstances(); i++) {
			Instance inst = dataEval.instance(i);
			double acv = dataEval.instance(i).classValue();
			String actual = dataEval.classAttribute().value((int) acv);
			double result = bayesNet.classifyInstance(inst);
			String prediction = dataEval.classAttribute().value((int) result);
			if (actual.compareTo(prediction) == 0) {
				numberOfCorrectClassification++;
			}
		}
		double result = numberOfCorrectClassification / numberOfInstances;
		
		
		return result;
	}

	static double getFitness(BayesNet bayesNet, Instances dataEval) {
		double numberOfInstances = dataEval.numInstances();
		double numberOfCorrectClassification = 0;
		for (int i = 0; i < dataEval.numInstances(); i++) {
			Instance inst = dataEval.instance(i);
			double acv = dataEval.instance(i).classValue();
			String actual = dataEval.classAttribute().value((int) acv);
			//System.out.println(inst);
			double result = 0;
			try {
				result = bayesNet.classifyInstance(inst);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				System.out.println(inst);
				e.printStackTrace();
			}
			String prediction = dataEval.classAttribute().value((int) result);
			// System.out.println("i: " + i + " ");
			// System.out.println("actual: " + actual);
			// System.out.println("prediction: " + prediction);
			if (actual.compareTo(prediction) == 0) {
				numberOfCorrectClassification++;
			}
		}
		double aic = bayesNet.measureAICScore();
		//aic += 500;
		//aic /= 10;
		//double result = numberOfCorrectClassification / numberOfInstances + aic;
		double result = numberOfCorrectClassification / numberOfInstances;
		
		return result;
	}

	private void setBayesNetStructureData() {
		numberOfNodes = 0;
		names = new ArrayList<String>();
		Enumeration<Attribute> atrr = data.enumerateAttributes();
		while (atrr.hasMoreElements()) {
			Attribute att = (Attribute) atrr.nextElement();
			names.add(att.name());
			//System.out.println(att.name());
			numberOfNodes++;
		}
		names.add("class");
		numberOfNodes++;
	}

	static Instances loadData(String path) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(path));
		ArffReader arff = new ArffReader(reader);
		Instances data = arff.getData();
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	private void createWekaBayesNet() throws Exception {
		BayesNet wekaBayesNet = new BayesNet();

		String[] netOptions = {"-D", "-Q", "weka.classifiers.bayes.net.search.local.SimulatedAnnealing", "--", "-A", "10.0", "-U", "10000", "-D", "0.999", "-R", "1", "-S", "BAYES", "-E", "weka.classifiers.bayes.net.estimate.SimpleEstimator", "--", "-A", "0.5"};
		wekaBayesNet.setOptions(netOptions);
		wekaBayesNet.buildClassifier(data);

		System.out.println("SimulatedAnnealing correct percentage: " + checkOnEvalData(wekaBayesNet, dataEval));
		//System.out.println("SimulatedAnnealing bayes score: " + wekaBayesNet.measureBayesScore());
		//System.out.println("SimulatedAnnealing aic score: " + wekaBayesNet.measureAICScore());
		//System.out.println("SimulatedAnnealing mdl score: " + wekaBayesNet.measureMDLScore());
		
		drawGraph(wekaBayesNet, "SimulatedAnnealing");
		System.out.println();
		/*
		 * String[] netOptions2 = {"-D", "-Q",
		 * "weka.classifiers.bayes.net.search.global.GeneticSearch", "--", "-L", "10",
		 * "-A", "100", "-U", "10", "-R", "1", "-M", "-C", "-S", "LOO-CV", "-E",
		 * "weka.classifiers.bayes.net.estimate.SimpleEstimator", "--", "-A", "0.5"};
		 * wekaBayesNet.setOptions(netOptions2); wekaBayesNet.buildClassifier(data);
		 * 
		 * System.out.println("weka correct percentage: " +
		 * checkOnEvalData(wekaBayesNet, dataEval));
		 * System.out.println("weka bayes score: " + wekaBayesNet.measureBayesScore());
		 * System.out.println("weka aic score: " + wekaBayesNet.measureAICScore());
		 * System.out.println("weka mdl score: " + wekaBayesNet.measureMDLScore());
		 * 
		 * drawGraph(wekaBayesNet, "wekaBayesNet graph");
		 */
		
		String[] netOptions3 = { "-D", "-Q", "weka.classifiers.bayes.net.search.global.TAN", "--", "-S", "LOO-CV", "-E",
				"weka.classifiers.bayes.net.estimate.SimpleEstimator", "--", "-A", "0.5" };
		wekaBayesNet.setOptions(netOptions3);
		wekaBayesNet.buildClassifier(data);

		System.out.println("TAN correct percentage: " + checkOnEvalData(wekaBayesNet, dataEval));
		//System.out.println("TAN bayes score: " + wekaBayesNet.measureBayesScore());
		//System.out.println("TAN aic score: " + wekaBayesNet.measureAICScore());
		//System.out.println("TAN mdl score: " + wekaBayesNet.measureMDLScore());
		
		drawGraph(wekaBayesNet, "TAN");
		System.out.println();
	}

	private static void drawGraph(BayesNet bayesNet, String name) throws BIFFormatException, Exception {
		GraphVisualizer gv = new GraphVisualizer();
		gv.doLayout();
		gv.readBIF(bayesNet.graph());
		JFrame jf = new JFrame(name);
		jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		jf.setSize(800, 600);
		jf.getContentPane().setLayout(new BorderLayout());
		jf.getContentPane().add(gv, BorderLayout.CENTER);
		jf.setVisible(true);
	}
}

class SortByFitness implements Comparator<MyBayesNet> {
	Instances dataEval;

	SortByFitness() throws IOException {
		dataEval = MyBayesNetGenerator.loadData(MyBayesNetGenerator.path[MyBayesNetGenerator.pathNumber] + "_eval.arff");
	}

	public int compare(MyBayesNet a, MyBayesNet b) {
		double result = 0;
		try {
			double result1 = MyBayesNetGenerator.getFitness(a.getBayesNet(), dataEval);
			double result2 = MyBayesNetGenerator.getFitness(b.getBayesNet(), dataEval);
			result = result2 - result1;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return (int) (result * 10000000);
	}
}
