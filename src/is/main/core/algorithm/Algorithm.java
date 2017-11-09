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
 * Algorithm.java
 * Copyright (C) 2010 Universidad de Burgos
 */

package main.core.algorithm;

import java.io.Serializable;
import java.util.Vector;

import main.core.exception.NotEnoughInstancesException;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.NearestNeighbourSearch;

/**
 * <b>Descripción</b><br>
 * Superclase para los algoritmos de Selección de Instancias.
 * <p>
 * <b>Detalles</b><br>
 * Posibilita la adición de nuevos algoritmos con una interfaz común.
 * Posibilita el seguimiento de las instancias seleccionadas a partir del índice sobre el dataset
 * original. 
 * </p>
 * <p>
 * <b>Funcionalidad</b><br>
 * Implementa estrategia en el patrón de diseño Strategy.
 * </p>
 * 
 * @author Álvar Arnaiz González
 * @version 1.2
 */
public abstract class Algorithm implements Serializable {
	
	/**
	 * Para la serialización
	 */
	private static final long serialVersionUID = -4056412941058900139L;

	/**
	 * Conjunto de entrenamiento del algoritmo.
	 */
	protected Instances mTrainSet;
	
	/**
	 * Conjunto solución para el algoritmo.
	 */
	protected Instances mSolutionSet;
	
	/**
	 * Instancia a evaluar. 
	 */
	protected Instance mCurrentInstance;
	
	/**
	 * Posición de la instancia actual.
	 */
	protected int mCurrInstancePos;
	
	/**
	 * Vector de índices para identificar cada instancia dentro del conjunto inicial de instancias.
	 * Numerando el conjunto inicial desde 0 hasta número de instancias - 1 en este atributo se
	 * almacena el índice de la instancia en el conjunto original.
	 */
	protected Vector<Integer> mInputDatasetIndex;
	
	/**
	 * Vector de índices para identificar cada instancia dentro del conjunto solución de instancias.
	 */
	protected Vector<Integer> mOutputDatasetIndex;
	
	/**
	 * Algoritmo de búsqueda de vecinos cercanos.
	 */
	protected NearestNeighbourSearch mNearestNeighbourSearch;
	
	/**
	 * Constructor por defecto del algoritmo de selección de instancias.
	 */
	public Algorithm () {
		mTrainSet = null;
		mSolutionSet = null;
		mCurrentInstance = null;
		mInputDatasetIndex = null;
		mOutputDatasetIndex = null;
		mCurrInstancePos = 0;
	} // Algorithm
	
	/**
	 * Constructor del algoritmo de selección de instancias al que se le pasa el nuevo conjunto
	 * de instancias a tratar.
	 * Es equivalente a:<br>
	 * <code>
	 * Algorithm();<br>
	 * reset(train);
	 * </code>
	 * 
	 * @param train Conjunto de instancias a seleccionar.
	 * @throws NotEnoughInstancesException Si el dataset no tiene instancias.
	 */
	public Algorithm (Instances train) throws NotEnoughInstancesException {
		this();
		
		reset(train);
	} // Algorithm
	
	/**
	 * Constructor del algoritmo de selección de instancias al que se le pasa el nuevo conjunto
	 * de instancias a tratar.
	 * Es equivalente a:<br>
	 * <code>
	 * Algorithm(train);<br>
	 * reset(inputDatasetIndex);
	 * </code>
	 * 
	 * @param train Conjunto de instancias a seleccionar.
	 * @param inputDatasetIndex Array de índices para identificar cada instancia dentro del conjunto
	 * inicial de instancias.
	 * @throws NotEnoughInstancesException Si el dataset no tiene instancias.
	 */
	public Algorithm (Instances train, int [] inputDatasetIndex) throws NotEnoughInstancesException {
		this();
		
		reset(train, inputDatasetIndex);
	} // Algorithm
	
	/**
	 * Devuelve el conjunto de entrenamiento del algoritmo. 
	 * 
	 * @return Conjunto de entrenamiento.
	 */
	public Instances getTrainSet () {
		
		return mTrainSet;
	} // getTrainSet
	
	/**
	 * Devuelve el conjunto solución del algoritmo. 
	 * 
	 * @return Conjunto reducido de instancias.
	 */
	public Instances getSolutionSet () {
		
		return mSolutionSet;
	} // getSolutionSet
	
	/**
	 * Devuelve la instancia a analizar en el siguiente paso del algoritmo.
	 * 
	 * @return Instancia a analizar.
	 */
	public Instance getCurrentInstance () {
		
		return mCurrentInstance;
	} // getCurrentInstance
	
	/**
	 * Devuelve el vector de índices del conjunto solución.
	 * 
	 * @return Vector de índices del conjunto solución seleccionado.
	 */
	public Vector<Integer> getOutputDatasetIndex () {
		
		return mOutputDatasetIndex;
	} // getOutputDatasetIndex
	
	/**
	 * Ejecuta un paso del algoritmo.
	 * 
	 * @return Verdadero si quedan pasos que ejecutar, falso en caso contratio.
	 * @throws Exception Excepción producida durante el paso del algoritmo.
	 */
	public abstract boolean step () throws Exception;
	
	/**
	 * Ejecuta todos los pasos del algoritmo desde el punto en el que se encuentre hasta el final.
	 * 
	 * @throws Exception Excepción producida durante el paso del algoritmo.
	 */
	public void allSteps () throws Exception {
		while (step ());
	} // allSteps
	
	/**
	 * Reinicia el algoritmo con un conjunto de entrenamiento nuevo.
	 * Inicializa las variables de trabajo del algoritmo.
	 *  
	 * @param train Conjunto de instancias a seleccionar.
	 * @throws NotEnoughInstancesException Si el dataset no tiene instancias.
	 */
	public void reset (Instances train) throws NotEnoughInstancesException {
		reset(train, getIntIndexArray(train.numInstances()));
	} // reset

	/**
	 * Reinicia el algoritmo con un conjunto de entrenamiento nuevo.
	 * Inicializa las variables de trabajo del algoritmo.
	 *  
	 * @param train Conjunto de instancias a seleccionar.
	 * @param inputDatasetIndex Array de índices para identificar cada instancia dentro del conjunto
	 * inicial de instancias.
	 * @throws NotEnoughInstancesException Si el dataset no tiene instancias.
	 */
	public void reset (Instances train, int[] inputDatasetIndex) throws NotEnoughInstancesException {
		// Error si el dataset no tiene suficientes instancias.
		if (train.numInstances() == 0)
			throw new NotEnoughInstancesException(NotEnoughInstancesException.MESSAGE);
		
		// Inicializar las variables del algoritmo.
		mTrainSet = train;
		mSolutionSet = new Instances(train, train.numInstances()/10);
		mInputDatasetIndex = new Vector<Integer>(inputDatasetIndex.length, 0);
		mOutputDatasetIndex = new Vector<Integer>(train.numInstances()/10);
		
		// Copiar inputDatsetIndex en mInputDatasetIndex
		for (int i = 0; i < inputDatasetIndex.length; i++)
			mInputDatasetIndex.add(inputDatasetIndex[i]);
	} // reset

	/**
	 * Comprueba si la instancia se clasifica correctamente por el conjunto de entrenamiento dado. 
	 * 
	 * @param instance Instancia a clasificar.
	 * @param trainSet Conjunto de vecinos de la instancia a evaluar.
	 * @return Verdadero si no se clasifica correctamente por sus vecinos, falso en caso contrario.
	 */
	protected boolean isMisclassified (Instance instance, Instances trainSet) {
		Vector<Instance> vTrainSet = new Vector<Instance>(trainSet.numInstances());
		
		for (int i = 0; i < trainSet.numInstances(); i++)
			vTrainSet.add(trainSet.instance(i));
		
		return isMisclassified(instance, vTrainSet);
	} // isMisclassified
	
	/**
	 * Comprueba si la instancia se clasifica correctamente por el conjunto de entrenamiento dado.
	 * En caso de empate devolverá false; 
	 * 
	 * @param instance Instancia a clasificar.
	 * @param trainSet Conjunto de vecinos de la instancia a evaluar.
	 * @return Verdadero si no se clasifica correctamente por sus vecinos, falso en caso contrario.
	 */
	protected boolean isMisclassified(Instance instance, Vector<Instance> trainSet) {
		Instance tmpInst;
		int numInstOtherClass, numInstSameClass = 0;
		
		// Contar el número de vecinos con la misma clase que instance.
		for (Instance inst : trainSet)
			if (inst.classValue() == instance.classValue())
				numInstSameClass++;
		
		// Recorrer el conjunto de entrenamiento.
		for (int i = 0; i < trainSet.size(); i++) {
			numInstOtherClass = 0;
			tmpInst = trainSet.elementAt(i);
			
			// Si la clase de tmpInst es distinta a la de la clase de instance, contar cuantas
			// instancias hay de esa clase.
			if (tmpInst.classValue() != instance.classValue()) {
				// Contar el numero de vecinos que tienen la clase de tmpInst.
				for (int j = i; j < trainSet.size(); j++)
					if (trainSet.elementAt(j).classValue() == tmpInst.classValue())
						numInstOtherClass++;
				
				// Si para la clase de tmpInst hay mas instancias que de la clase instance devolver
				// true, sino seguir mirando el resto de instancias.
				if (numInstOtherClass > numInstSameClass)
					return true;
			}
		}
		
		return false;
	} // isMisclassified
	
	/**
	 * Crea un vector de índices de tamaño size.
	 * El primer elemento del vector contiene 0, el siguiente 1 y así sucesivamente hasta size - 1.
	 * 
	 * @param size Número de elementos del vector.
	 * @return Vector de índices.
	 */
	protected int[] getIntIndexArray (int size) {
		int[] indexOfInstances = new int[size];
		
		for (int i = 0; i < size; i++)
			indexOfInstances[i] = i;

		return indexOfInstances;
	} // getIntIndexArray

	/**
	 * Devuelve el vector de enteros como array.
	 * 
	 * @param vector Vector a transformar.
	 * @return Array de enteros con la misma información que vector.
	 */
	protected int[] vectorToArray (Vector<Integer> vector) {
		int array[] = new int[vector.size()];
		
		for (int i = 0; i < vector.size(); i++)
			array[i] = vector.elementAt(i);
		
		return array;
	} // vectorToArray

} // Algorithm
