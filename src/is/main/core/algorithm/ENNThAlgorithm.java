/*
 * This file is part of Instance Selection Library.
 * 
 * Instance Selection Library is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Instance Selection Library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Instance Selection Library.  If not, see <http://www.gnu.org/licenses/>.
 * 
 * ENNThAlgorithm.java
 * Copyright (C) 2015 Universidad de Burgos
 */

package main.core.algorithm;

import java.io.Serializable;

import main.core.exception.NotEnoughInstancesException;
import main.core.util.LinearISNNSearch;
import weka.core.Instances;

/**
 * <b>Descripción</b><br>
 * Algoritmo ENNth. Presentado en: Sánchez, José Salvador, et al. "Analysis of
 * new techniques to obtain quality training sets." Pattern Recognition Letters
 * 24.7 (2003): 1015-1022.
 * <p>
 * <b>Detalles</b><br>
 * Utilizado el código de Keel.
 * </p>
 * <p>
 * <p>
 * <b>Funcionalidad</b><br>
 * Implementa el algoritmo ENNth.
 * </p>
 * 
 * @author Álvar Arnaiz González
 * @version 1.5
 */
public class ENNThAlgorithm extends Algorithm implements Serializable {

	/**
	 * Para la serialización.
	 */
	private static final long serialVersionUID = 5738067038422735191L;

	/**
	 * Número de vecinos cercanos a buscar.
	 */
	private int mNumOfNearestNeighbour;

	/**
	 * Umbral: entre 0 y 1.
	 */
	private double mMu;

	/**
	 * Constructor por defecto del algoritmo ENN. Antes de comenzar la ejecución
	 * del algoritmo debe llamarse a setNumOfNearestNeighbour para establecer el
	 * número de vecinos cercanos deseado, por defecto es 1.
	 */
	public ENNThAlgorithm() {
		super();

		// Por defecto el número de vecinos cercanos es 1.
		mNumOfNearestNeighbour = 1;
		// Por defecto mu = 0.7
		mMu = 0.7;
	} // ENNThAlgorithm

	/**
	 * Constructor del algoritmo ENN al que se le pasa el nuevo conjunto de
	 * instancias a tratar. Antes de comenzar la ejecución del algoritmo debe
	 * llamarse a setNumOfNearestNeighbour para establecer el número de vecinos
	 * cercanos deseado, por defecto es 1.<br>
	 * Es equivalente a:<br>
	 * <code>
	 * Algorithm();<br>
	 * reset(train);
	 * </code>
	 * 
	 * @param train
	 *            Conjunto de instancias a seleccionar.
	 * @throws NotEnoughInstancesException
	 *             Si el dataset no tiene instancias.
	 */
	public ENNThAlgorithm(Instances train) throws NotEnoughInstancesException {
		super(train);

		// Por defecto el número de vecinos cercanos es 1.
		mNumOfNearestNeighbour = 1;
		// Por defecto mu = 0.7
		mMu = 0.7;
	} // ENNThAlgorithm

	/**
	 * Constructor del algoritmo ENN al que se le pasa el nuevo conjunto de
	 * instancias a tratar. Antes de comenzar la ejecución del algoritmo debe
	 * llamarse a setNumOfNearestNeighbour para establecer el número de vecinos
	 * cercanos deseado, por defecto es 1.<br>
	 * Es equivalente a:<br>
	 * <code>
	 * Algorithm(train);<br>
	 * reset(inputDatasetIndex);
	 * </code>
	 * 
	 * @param train
	 *            Conjunto de instancias a seleccionar.
	 * @param inputDatasetIndex
	 *            Array de índices para identificar cada instancia dentro del
	 *            conjunto inicial de instancias.
	 * @throws NotEnoughInstancesException
	 *             Si el dataset no tiene instancias.
	 */
	public ENNThAlgorithm(Instances train, int[] inputDatasetIndex)
			throws NotEnoughInstancesException {
		super(train, inputDatasetIndex);

		// Por defecto el número de vecinos cercanos es 1.
		mNumOfNearestNeighbour = 1;
		// Por defecto mu = 0.7
		mMu = 0.7;
	} // ENNThAlgorithm

	/**
	 * Devuelve el número de vecinos cercanos que va a utilizar el algoritmo.
	 * 
	 * @return Número de vecinos cercanos a utilizar por el algoritmo.
	 */
	public int getNumOfNearestNeighbour() {

		return mNumOfNearestNeighbour;
	} // getNumOfNearestNeighbour

	/**
	 * Establece el valor de Mu.
	 * 
	 * @param mu Umbral.
	 * @throws IllegalArgumentException
	 *             Es lanzada si el umbral está fuera del intervalo [0, 1]
	 */
	public void setMu (double mu) {
		if (mu <= 0.0 || mu >= 1.0)
			throw new IllegalArgumentException("Mu debe estar en el intervalo [0, 1]");

		mMu = mu;
	} // setMu

	/**
	 * Devuelve el valor de Mu.
	 * 
	 * @return Valor de Mu.
	 */
	public double getMu () {

		return mMu;
	} // getMu

	/**
	 * Establece el número de vecinos cercanos que va a utilizar el algoritmo.
	 * 
	 * @param nn
	 *            Número de vecinos cercanos a utilizar por el algoritmo.
	 * @throws IllegalArgumentException
	 *             Es lanzada si el número de vecinos es menor que 1.
	 */
	public void setNumOfNearestNeighbour(int nn) {
		if (nn < 1)
			throw new IllegalArgumentException(
					"El número de vecinos cercanos debe ser mayor de 0.");

		mNumOfNearestNeighbour = nn;
	} // setNumOfNearestNeighbour

	/**
	 * Ejecuta un paso del algoritmo. Comienza con un conjunto vacío.
	 * 
	 * @return Verdadero si quedan pasos que ejecutar, falso en caso contratio.
	 * @throws Exception Excepción producida durante el paso del algoritmo.
	 */
	public boolean step() throws Exception {
		Instances nearestNeighbours;
		int predClass;
		double prob[];
		double sumProb;
		double maxProb;
		int j, pos;

		/*
		 * Body of the algorithm. For each instance in T, search the correspond
		 * class conform his mayority from the nearest neighborhood. Is it is
		 * positive, the instance is selected.
		 */
		nearestNeighbours = mNearestNeighbourSearch.kNearestNeighbours(mCurrentInstance, mNumOfNearestNeighbour);

		prob = new double[mTrainSet.numClasses()];

		for (j = 0; j < nearestNeighbours.numInstances(); j++)
			prob[(int) nearestNeighbours.instance(j).classValue()] += 1.0 / (1.0 + 
			  mNearestNeighbourSearch.getDistanceFunction().distance(mCurrentInstance, nearestNeighbours.instance(j)));
		
		sumProb = 0.0;
		
		for (j = 0; j < prob.length; j++)
			sumProb += prob[j];
		
		for (j = 0; j < prob.length; j++)
			prob[j] /= sumProb;

		maxProb = prob[0];
		pos = 0;
		
		for (j = 1; j < prob.length; j++)
			if (prob[j] > maxProb) {
				maxProb = prob[j];
				pos = j;
			}

		predClass = pos;
		
		// agree with your majority, it is included in the solution set
		if (predClass == mCurrentInstance.classValue() && maxProb > mMu) { 
			mSolutionSet.add(mCurrentInstance);
			mOutputDatasetIndex.add(mInputDatasetIndex.get(mCurrInstancePos));
		}

		mCurrInstancePos++;
		
		// Si se han procesado todas las intancias finalizar.
		if (mTrainSet.numInstances() == mCurrInstancePos)
			return false;

		// Pasar a la siguiente instancia.
		mCurrentInstance = mTrainSet.instance(mCurrInstancePos);

		return true;
	} // step

	/**
	 * Reinicia el algoritmo con un conjunto de entrenamiento nuevo. Inicializa
	 * las variables de trabajo del algoritmo. Antes de comenzar la ejecución
	 * del algoritmo debe llamarse a setNumOfNearestNeighbour para establecer el
	 * número de vecinos cercanos deseado, por defecto es 1.
	 * 
	 * @param train
	 *            Conjunto de entrenamiento.
	 * @throws NotEnoughInstancesException
	 *             Si el dataset no tiene instancias.
	 */
	public void reset(Instances train) throws NotEnoughInstancesException {
		super.reset(train);
	} // reset

	/**
	 * Reinicia el algoritmo con un conjunto de entrenamiento nuevo. Inicializa
	 * las variables de trabajo del algoritmo. Antes de comenzar la ejecución
	 * del algoritmo debe llamarse a setNumOfNearestNeighbour para establecer el
	 * número de vecinos cercanos deseado, por defecto es 1.
	 * 
	 * @param train
	 *            Conjunto de entrenamiento.
	 * @param inputDatasetIndex
	 *            Array de índices a las instancias del dataset.
	 * @throws NotEnoughInstancesException
	 *             Si el dataset no tiene instancias.
	 */
	public void reset(Instances train, int[] inputDatasetIndex) throws NotEnoughInstancesException {
		super.reset(train, inputDatasetIndex);

		init();
	} // reset

	/**
	 * Inicializa las variables del algoritmo.
	 */
	private void init() {
		// Inicializar la instancia actual.
		mCurrentInstance = mTrainSet.firstInstance();
		mCurrInstancePos = 0;

		// Inicializar el algoritmo de vecinos cercanos.
		mNearestNeighbourSearch = new LinearISNNSearch(mTrainSet);
	} // init

} // ENNThAlgorithm
