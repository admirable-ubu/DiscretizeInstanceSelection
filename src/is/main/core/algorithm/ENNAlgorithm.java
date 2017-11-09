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
 * ENNAlgorithm.java
 * Copyright (C) 2010 Universidad de Burgos
 */

package main.core.algorithm;

import java.io.Serializable;

import main.core.exception.NotEnoughInstancesException;
import main.core.util.LinearISNNSearch;
import weka.core.Instances;

/**
 * <b>Descripción</b><br>
 * Algoritmo del vecino mas cercano editado.
 * <p>
 * <b>Detalles</b><br>
 * Su criterio de selección se basa en eliminar o descartar el objeto en cuestión si la clase de 
 * éste no coincide con la de la mayoría de sus k vecinos más cercanos.  
 * Posibilita seguir al algoritmo paso a paso.  
 * </p>
 * <p>
 * <b>Pseudocódigo del ENN</b><br>
 * <span style="font-weight: bold;">Require:</span> Training set <span style="font-style: italic;">X</span> = 
 * {(x<sub>1</sub>, y<sub>1</sub>),...,(x<sub>n</sub>, y<sub>n</sub>)}, <span style="font-style: italic;">k
 * </span>&nbsp; number of nearest neighbors<br><span style="font-weight: bold;">Ensure: </span>Editing Subset 
 * <span style="font-style: italic;">S&nbsp;&#8834; X</span><br><br>1: <span style="font-style: italic;">S = X
 * </span><br>2: <span style="font-weight: bold;">for all</span> x &#8712; <span style="font-style: italic;">S
 * </span> <span style="font-weight: bold;">do</span><br>3:&nbsp;&nbsp;&nbsp; <span style="font-weight: bold;">
 * find</span> <span style="font-style: italic;">x.N</span><sub style="font-style: italic;">1...k+1</sub>, the
 *  k + 1 nearest neighbors of <span style="font-style: italic;">x</span> in <span style="font-style: italic;">
 *  X - {x}</span><br>4:&nbsp;&nbsp;&nbsp; <span style="font-weight: bold;">if</span>&nbsp;
 *  <span style="font-style: italic;">&#948;</span><sub style="font-style: italic;">kNN</sub>
 *  <span style="font-style: italic;">(x</span><sub style="font-style: italic;">i</sub>
 *  <span style="font-style: italic;">)&nbsp;&#8800;&nbsp;&#920;</span><sub style="font-style: italic;">i
 *  </sub> <span style="font-weight: bold;">then<br></span>5:&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; Remove
 *  &nbsp;<span style="font-style: italic;">x</span>&nbsp;of <span style="font-style: italic;">S</span><br>
 *  6:&nbsp;&nbsp;&nbsp; <span style="font-weight: bold;">end if</span><br>7: <span style="font-weight: bold;">
 *  end for</span><br>8:<span style="font-weight: bold;"> return</span> <span style="font-style: italic;">S</span>
 * </p>
 * <p>
 * <b>Funcionalidad</b><br>
 * Implementa el algoritmo ENN.
 * </p>
 * 
 * @author Álvar Arnaiz González
 * @version 1.2
 */
public class ENNAlgorithm extends Algorithm implements Serializable {
	
	/**
	 * Para la serialización.
	 */
	private static final long serialVersionUID = 5738067038422735191L;

	/**
	 * Número de vecinos cercanos a buscar.
	 */
	private int mNumOfNearestNeighbour;
	
	/**
	 * Número de pasos que el algoritmo lleva ejecutados. 
	 */
	private int mNumOfIterations;
	
	/**
	 * Constructor por defecto del algoritmo ENN.
	 * Antes de comenzar la ejecución del algoritmo debe llamarse a setNumOfNearestNeighbour para
	 * establecer el número de vecinos cercanos deseado, por defecto es 1.
	 */
	public ENNAlgorithm () {
		super();
		
		// Por defecto el número de vecinos cercanos es 1.
		mNumOfNearestNeighbour = 1;
	} // ENNAlgorithm
	
	/**
	 * Constructor del algoritmo ENN al que se le pasa el nuevo conjunto de instancias a tratar.
	 * Antes de comenzar la ejecución del algoritmo debe llamarse a setNumOfNearestNeighbour para
	 * establecer el número de vecinos cercanos deseado, por defecto es 1.<br>
	 * Es equivalente a:<br>
	 * <code>
	 * Algorithm();<br>
	 * reset(train);
	 * </code>
	 * 
	 * @param train Conjunto de instancias a seleccionar.
	 * @throws NotEnoughInstancesException Si el dataset no tiene instancias.
	 */
	public ENNAlgorithm (Instances train) throws NotEnoughInstancesException {
		super(train);
		
		// Por defecto el número de vecinos cercanos es 1.
		mNumOfNearestNeighbour = 1;
	} // ENNAlgorithm
	
	/**
	 * Constructor del algoritmo ENN al que se le pasa el nuevo conjunto de instancias a tratar.
	 * Antes de comenzar la ejecución del algoritmo debe llamarse a setNumOfNearestNeighbour para
	 * establecer el número de vecinos cercanos deseado, por defecto es 1.<br>
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
	public ENNAlgorithm (Instances train, int [] inputDatasetIndex) throws NotEnoughInstancesException {
		super(train, inputDatasetIndex);
		
		// Por defecto el número de vecinos cercanos es 1.
		mNumOfNearestNeighbour = 1;
	} // ENNAlgorithm
	
	/**
	 * Devuelve el número de vecinos cercanos que va a utilizar el algoritmo. 
	 * 
	 * @return Número de vecinos cercanos a utilizar por el algoritmo.
	 */
	public int getNumOfNearestNeighbour () {
		
		return mNumOfNearestNeighbour;
	} // getNumOfNearestNeighbour
	
	/**
	 * Establece el número de vecinos cercanos que va a utilizar el algoritmo. 
	 * 
	 * @param nn Número de vecinos cercanos a utilizar por el algoritmo.
	 * @throws IllegalArgumentException Es lanzada si el número de vecinos es menor que 1.
	 */
	public void setNumOfNearestNeighbour (int nn) {
		if (nn < 1)
			throw new IllegalArgumentException("El número de vecinos cercanos debe ser mayor de 0.");
		
		mNumOfNearestNeighbour = nn;
	} // setNumOfNearestNeighbour

	/**
	 * Ejecuta un paso del algoritmo.
	 * Comienza con un conjunto solución que tiene todas las instancias del dataset original.
	 * Recorre el conjunto solución comprobando cada instancia si se clasifica correctamente o no.
	 * Si no se clasifica incorrectamente la elimina del conjunto solución.
	 * 
	 * @return Verdadero si quedan pasos que ejecutar, falso en caso contratio.
	 * @throws Exception Excepción producida durante el paso del algoritmo.
	 */
	public boolean step () throws Exception {
		// Aumentar el número de iteraciones.
		mNumOfIterations++;
		
		// Si no se clasifica correctamente por los vecinos cercanos se elimina.
		if (isMisclassified(mCurrentInstance, 
		      mNearestNeighbourSearch.kNearestNeighbours(mCurrentInstance, mNumOfNearestNeighbour))) {
			mSolutionSet.delete(mCurrInstancePos);
			mOutputDatasetIndex.remove(mCurrInstancePos);
		}
		else {
			mCurrInstancePos++;
		}
		
		// Si se han procesado todas las intancias finalizar.
		if (mTrainSet.numInstances() == mNumOfIterations)
			return false;
		
		// Pasar a la siguiente instancia.
		mCurrentInstance = mSolutionSet.instance(mCurrInstancePos);
		
		return true;
	} // step
	
	/**
	 * Reinicia el algoritmo con un conjunto de entrenamiento nuevo.
	 * Inicializa las variables de trabajo del algoritmo.
	 * Antes de comenzar la ejecución del algoritmo debe llamarse a setNumOfNearestNeighbour para
	 * establecer el número de vecinos cercanos deseado, por defecto es 1.
	 *  
	 * @param train Conjunto de entrenamiento.
	 * @throws NotEnoughInstancesException Si el dataset no tiene instancias.
	 */
	public void reset (Instances train) throws NotEnoughInstancesException {
		super.reset(train);
	} // reset

	/**
	 * Reinicia el algoritmo con un conjunto de entrenamiento nuevo.
	 * Inicializa las variables de trabajo del algoritmo.
	 * Antes de comenzar la ejecución del algoritmo debe llamarse a setNumOfNearestNeighbour para
	 * establecer el número de vecinos cercanos deseado, por defecto es 1.
	 *  
	 * @param train Conjunto de entrenamiento.
	 * @param inputDatasetIndex Array de índices a las instancias del dataset. 
	 * @throws NotEnoughInstancesException Si el dataset no tiene instancias.
	 */
	public void reset (Instances train, int[] inputDatasetIndex) throws NotEnoughInstancesException {
		super.reset(train, inputDatasetIndex);

		init();
	} // reset

	/**
	 * Inicializa las variables del algoritmo.
	 */
	private void init () {
		// Inicializar el número de iteraciones.
		mNumOfIterations = 0;
		
		// Copiar el conjunto de entrenamiento en el conjunto solución.
		mSolutionSet = new Instances(mTrainSet);
		
		// Inicializar el vector de índices de salida.
		for (Integer index : mInputDatasetIndex)
			mOutputDatasetIndex.add(new Integer(index));
		
		// Inicializar la instancia actual.
		mCurrentInstance = mSolutionSet.firstInstance();
		mCurrInstancePos = 0;
		
		// Inicializar el algoritmo de vecinos cercanos.
		mNearestNeighbourSearch = new LinearISNNSearch(mTrainSet);
	} // init
	
} // ENNAlgorithm
