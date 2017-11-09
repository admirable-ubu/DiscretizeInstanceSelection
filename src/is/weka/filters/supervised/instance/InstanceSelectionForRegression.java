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
 * InstanceSelectionForRegression.java
 * Copyright (C) 2016 Universidad de Burgos
 */

package weka.filters.supervised.instance;

import main.core.algorithm.Algorithm;
import main.core.algorithm.ENNRegAlgorithm;
import main.core.algorithm.MIAlgorithm;
import main.core.exception.NotEnoughInstancesException;

import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.Enumeration;
import java.util.Vector;

/**
 * <b>Descripción</b><br>
 * Filtro que implementa algoritmos de selección de instancias para regresión.
 * <p>
 * <b>Detalles</b><br>
 * Implementa los siguientes algoritmos:
 * <ul>
 *  <li> MI: Algoritmo basado en la información mutua.
 *  <li> Reg-ENN: ENN para regresión.
 * </ul>  
 * <b>Funcionalidad</b><br>
 * Utiliza la biblioteca de algoritmos de selección de instancias realizada para el proyecto de final
 * de carrera en la Universidad de Burgos. Tutelado por: César García Osorio y Juan José Rodríguez Díez.
 * <p>
 * 
 * @author Álvar Arnaiz González
 * @version 1.6
 */
public class InstanceSelectionForRegression extends Filter implements SupervisedFilter, OptionHandler, InstanceSelectionFilterIF {

	/**
	 * Serial UID.
	 */
	private static final long serialVersionUID = 8190462201379437586L;

	/**
	 * Algoritmo de selección de instancias a utilizar.
	 */
	private main.core.algorithm.Algorithm mAlgorithm;
	
	/**
	 * Tiempo de CPU utilizado en el filtrado.
	 */
	private long mCPUTimeElapsed;
	
	/**
	 * Tiempo utilizado en el filtrado.
	 */
	private long mUserTimeElapsed;
	
	/**
	 * Número de vecinos cercanos a utilizar.
	 */
	private int mNearestNeighbourNum = 1;
	
	/**
	 * Valor de corte alfa.
	 */
	private double mAlpha = 0.5;
	
	/**
	 * Valor de corte beta.
	 */
	private double mBeta = 5.0;
	
	/**
	 * Algoritmo MI.
	 */
	public static final int TYPE_MI = 0;
	
	/**
	 * Algoritmo MI.
	 */
	public static final int TYPE_MI_SUMMATION = 1;
	
	/**
	 * Algoritmo RegENN.
	 */
	public static final int TYPE_REG_ENN = 2;
	
	/**
	 * Algoritmo RegCNN.
	 */
	public static final int TYPE_REG_CNN = 3;
	
	/**
	 * Algoritmo RegMSS.
	 */
	public static final int TYPE_REG_MSS = 4;
	
	/**
	 * Algoritmo RegICF.
	 */
	public static final int TYPE_REG_ICF = 5;
	
	/**
	 * Algoritmo RegLSS.
	 */
	public static final int TYPE_REG_LSS = 6;
	
	/**
	 * Algoritmo RegLSB.
	 */
	public static final int TYPE_REG_LSB = 7;
	
	/**
	 * Algoritmos implementados.
	 */
	public static final Tag[] TAGS_TYPE = {new Tag (TYPE_MI, "MI Algorithm"),
	                                       new Tag (TYPE_REG_ENN, "Reg ENN")};
	
	/**
	 * Algoritmo por defecto: MI.
	 */
	protected int mType = TYPE_MI;
	
	/**
	 * Constructor por defecto.
	 */
	public InstanceSelectionForRegression () {
	} // InstanceSelectionForRegression
	
	/**
	 * Devuelve el número de vecinos a utilizar.
	 * 
	 * @return Número de vecinos cercanos a utilizar.
	 */
	public int getNumOfNearestNeighbour () {
		
		return mNearestNeighbourNum;
	} // getNumOfNearestNeighbour
	
	/**
	 * Devuelve el valor de corte alfa.
	 * 
	 * @return Valor de alfa.
	 */
	public double getAlpha () {
		
		return mAlpha;
	} // getAlpha
	
	/**
	 * Devuelve el valor de corte beta.
	 * 
	 * @return Valor de beta.
	 */
	public double getBeta () {
		
		return mBeta;
	} // getBeta
	
	/**
	 * Establece el número de vecinos a utilizar.
	 * 
	 * @param nn Número de vecinos cercanos a utilizar.
	 * @throws IllegalArgumentException Es lanzada si el número de vecinos es menor que 1.
	 */
	public void setNumOfNearestNeighbour (int nn) {
		mNearestNeighbourNum = nn;
	} // setNumOfNearestNeighbour
	
	/**
	 * Returns the tip text for this property.
	 * 
	 * @return Nearest neighbour number to use.
	 */
	public String numOfNearestNeighbourTipText () {
		
		return "Nearest neighbour number to used.";
	} // numOfNearestNeighbourTipText

	/**
	 * Establece el valor de corte alfa.
	 * 
	 * @param alpha Valor alfa.
	 */
	public void setAlpha (double alpha) {
		mAlpha = alpha;
	} // setAlpha

	/**
	 * Establece el valor de corte beta.
	 * 
	 * @param alpha Valor beta.
	 */
	public void setBeta (double beta) {
		mBeta = beta;
	} // setBeta

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return Alpha value to use in algorithm. For IS.
	 */
	public String alphaTipText () {
		
		return "Alpha value to use in algorithm (For instance selection).";
	} // alphaTipText

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return Beta value to use in algorithm. For filtering.
	 */
	public String betaTipText () {
		
		return "Beta value to use in algorithm (For filtering stage).";
	} // betaTipText

	/**
	 * Devuelve las opciones del algoritmo.
	 * 
	 * @return Parámetros de algoritmo.
	 */
	public String[] getOptions () {
		Vector<String> result = new Vector<String>();
		 
		result.add("-K");
		result.add("" + getNumOfNearestNeighbour());
		    
		result.add("-A");
		result.add("" + getAlpha());
		    
		result.add("-B");
		result.add("" + getBeta());
		    
		result.add("-T");
		result.add("" + mType);
		 			    
		return result.toArray(new String[result.size()]); 
	} // getOptions

	/**
	 * Establece el tipo de algoritmo de selección de instancias a utilizar.
	 * 
	 * @param value Algoritmo seleccionado.
	 */
	public void setType (SelectedTag value) {
		if (value.getTags() == TAGS_TYPE)
			mType = value.getSelectedTag().getID();
	} // setSVMType

	/**
	 * Deveuelve el algoritmo de selección de instancias a utilizar.
	 * 
	 * @return Algoritmo seleccionado.
	 */
	public SelectedTag getType () {
		
		return new SelectedTag(mType, TAGS_TYPE);
	} // getSVMType
	
	/**
	 * Returns the tip text for this property.
	 * 
	 * @return Instance selection algorithm to use.
	 */
	public String typeTipText () {
		
		return "Instance selection algorithm to use.";
	} // typeTipText
	
	/**
	 * Lista los parámetros del algoritmo.
	 * 
	 * @return Parámetros del algoritmo.
	 */
	public Enumeration<Option> listOptions () {
		Vector<Option> newVector = new Vector<Option>(3);

		newVector.addElement(new Option("\tSpecifies the number of nearest neighbours\n" +
		                                "\t(default 1)", "K", 1, "-K <num>"));
		
		newVector.addElement(new Option("\tSpecifies alpha value\n" +
		                                "\t(default 0.5)", "A", 1, "-A <num>"));
		
		newVector.addElement(new Option("\tSpecifies beta value\n" +
		                                "\t(default 5)", "B", 1, "-B <num>"));

		newVector.addElement(new Option("\tSet type of solver (default: 0)\n"+
		                                "\t\t 0 = MI Algorithm\n"+
		                                "\t\t 2 = Reg ENN\n",
		                                "T", 1, "-T <int>"));

		return newVector.elements();
	} // listOptions

	/**
	 * Establece los parámetros/opciones del algoritmo.
	 * 
	 * @param options Opciones/Parámetros del algoritmo.
	 */
	public void setOptions (String[] options) throws Exception {
		String numStr = Utils.getOption('K', options);
		String tmpStrA = Utils.getOption('A', options);
		String tmpStrB = Utils.getOption('B', options);
		String typeStr = Utils.getOption('T', options);
		
		// Si el número de vecinos cercanos es distinto de 0 se asigna, sino se utilizará 1.
		if (numStr.length() != 0)
			setNumOfNearestNeighbour(Integer.parseInt(numStr));
		else
	    	setNumOfNearestNeighbour(1);
	    
		// Si el valor de alfa está entre 0 y 100 se asigna, sino se utilizará 0.5.
	    if (tmpStrA.length() != 0)
	    	setAlpha(Double.parseDouble(tmpStrA));
	    else
	    	setAlpha(0.5);

		// Si el valor de alfa está entre 0 y 100 se asigna, sino se utilizará 5.
	    if (tmpStrB.length() != 0)
	    	setBeta(Double.parseDouble(tmpStrB));
	    else
	    	setBeta(5.0);

		// Si el tipo de algoritmo es distinto de 0 se asigna, sino se utilizará del CNN.
	    if (typeStr.length() != 0)
	    	setType(new SelectedTag(Integer.parseInt(typeStr), TAGS_TYPE));
	    else
	    	setType(new SelectedTag(TYPE_MI, TAGS_TYPE));
	} // setOptions

	/**
	 * Returns default capabilities of the classifier.
	 * 
	 * @return the capabilities of this classifier
	 */
	public Capabilities getCapabilities () {
		Capabilities result = super.getCapabilities();

		// class
		result.disable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.DATE_CLASS);

		return result;
	} // getCapabilities

	/**
	 * Establece el formato de entrada de las instancias.
	 * 
	 * @param instanceInfo Una colección de instancias que contiene la estructura de entrada (cualquier
	 * instancia contenida en el conjunto será ignorada; solo se utilizará la estructura/cabecera).
	 * @return Verdadero si contiene una estructra válida.
	 * @throws Exception Si el formato de entrada no ha podido ser establecido satisfactoriamente.
	 */
	public boolean setInputFormat (Instances instanceInfo) throws Exception {
		super.setInputFormat(instanceInfo);
		super.setOutputFormat(instanceInfo);
	    
		return true;
	} // setInputFormat

	/**
	 * Introduce una nueva instancia al filtro.
	 * El filtro requiere que todas las instancias de entrenamiento sean leídas antes de producir la salida.
	 *
	 * @param instance Instancia de entrada.
	 * @return Verdadero si la instancia puede ser introducida al filtro.
	 * @throws IllegalStateException Si no se ha definido la estructura de entrada de las instancias.
	 */
	public boolean input (Instance instance) {
		if (getInputFormat() == null)
			throw new IllegalStateException("No input instance format defined");
		
		// Si es un nuevo batch.
		if (m_NewBatch) {
			resetQueue();
			m_NewBatch = false;
		}
		
		// Si se ha realizado el primer batch.
		if (m_FirstBatchDone) {
			push(instance);
			return true;
		}
		else {
			bufferInput(instance);
			return false;
		}
	} // input

	/**
	 * Establece que el filtro debe finalizar la entrada de instancias.
	 * Procesará las instancias almacenadas para devolver, mediante <code>output()</code> las instancias
	 * seleccionadas tras el filtrado.
	 *
	 * @return Verdadero si las instancias han sido procesadas.
	 * @throws IllegalStateException Si no se ha definido la estructura de las instancias.
	 * @throws Exception Si el algoritmo devuelve algun error durante la selección de instancias.
	 */
	public boolean batchFinished () throws Exception {
		// Si no se dispone de la cabecera.
		if (getInputFormat() == null)
			throw new IllegalStateException("No input instance format defined");
		
		// Realizar la selección de instancias.
		if (!m_FirstBatchDone)
			filter(getInputFormat());
		
		flushInput();
		
		m_NewBatch = true;
		m_FirstBatchDone = true;
		
		return (numPendingOutput() != 0);
	} // batchFinished
	
	/**
	 * Realiza la selección de instancias.
	 * Las instancias se añadirán a una cola.
	 * 
	 * @param inst Instancias a filtrar.
	 * @throws Exception Si el algoritmo ha producido algún error durante su ejecución.
	 */
	public void filter (Instances inst) throws Exception {
		ThreadMXBean thMonitor = ManagementFactory.getThreadMXBean();
		Instances solution;
		boolean canMeasureCPUTime = thMonitor.isThreadCpuTimeSupported();
		
		// Si se puede medir la CPU
		if(canMeasureCPUTime && !thMonitor.isThreadCpuTimeEnabled())
			thMonitor.setThreadCpuTimeEnabled(true);
		
		long thID = Thread.currentThread().getId();
		long CPUStartTime=-1, userTimeStart;
		
		userTimeStart = System.currentTimeMillis();
		
		if(canMeasureCPUTime)
			CPUStartTime = thMonitor.getThreadUserTime(thID);

		// Crear el algoritmo de selección de instancias y asignar el número de vecinos (si es necesario).
		try {
			if (mType == TYPE_MI) {
				mAlgorithm = new MIAlgorithm(inst);
				((MIAlgorithm)mAlgorithm).setNumOfNearestNeighbour(mNearestNeighbourNum);
				((MIAlgorithm)mAlgorithm).setAlpha(mAlpha);
			} else if (mType == TYPE_REG_ENN) {
				mAlgorithm = new ENNRegAlgorithm(inst);
				((ENNRegAlgorithm)mAlgorithm).setNumOfNearestNeighbour(mNearestNeighbourNum);
				((ENNRegAlgorithm)mAlgorithm).setAlpha(mAlpha);
			}
		}catch (NotEnoughInstancesException ex) {
			ex.printStackTrace();
			throw new IllegalStateException("The dataset has not enough instances");
		} catch (IllegalArgumentException ex) {
			throw new Exception("Neighbour number, alpha or beta value is wrong");
		} catch (Exception ex) {
			ex.printStackTrace();
			throw new Exception("Invalid Algorithm");
		}
		
		// Si el algoritmo existe, ejecutar todos sus pasos.
		if (mAlgorithm != null)
			mAlgorithm.allSteps();
		
		if(canMeasureCPUTime)
			mCPUTimeElapsed = (thMonitor.getThreadUserTime(thID) - CPUStartTime) / 1000000;
		
		mUserTimeElapsed = System.currentTimeMillis() - userTimeStart;
		
		thMonitor = null;
		
		solution = mAlgorithm.getSolutionSet();
		
		// Introducir en la cola las instancias devueltas por el algoritmo.
		for(int i=0; i<solution.numInstances(); i++)
			push(solution.instance(i));
	} // filter	  
	
	/**
	 * Devuelve el algoritmo de selección de instancias.
	 * 
	 * @return Algoritmo de selección de instancias.
	 */
	public Algorithm getAlgorithm () {
		
		return mAlgorithm;
	} // getAlgorithm

	/**
	 * Devuelve el conjunto de instancias devuelto por el algoritmo.
	 * 
	 * @return Conjunto de instancias solución.
	 */
	public Instances getSolutionSet() {
		
		return mAlgorithm.getSolutionSet();
	} // getSolutionSet
	
	/**
	 * Devuelve el tiempo de CPU empleado en el filtrado por el algoritmo de selección de instancias 
	 * seleccionado.
	 * 
	 * @return Tiempo de CPU utilizado en el filtrado.
	 */
	public long getFilterCPUTime () {
	
		return mCPUTimeElapsed;
	} // getFilterCPUTime
	
	/**
	 * Devuelve el tiempo empleado en el filtrado por el algoritmo de selección de instancias 
	 * seleccionado.
	 * 
	 * @return Tiempo utilizado en el filtrado.
	 */
	public long getFilterUserTime () {
	
		return mUserTimeElapsed;
	} // getFilterUserTime

} // InstanceSelectionForRegression
