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
 * InstanceSelection.java
 * Copyright (C) 2010 Universidad de Burgos
 */

package weka.filters.supervised.instance;

import main.core.algorithm.Algorithm;
import main.core.algorithm.ENNAlgorithm;
import main.core.algorithm.ENNThAlgorithm;
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
 * Filtro que implementa algoritmos de selección de instancias.
 * <p>
 * <b>Detalles</b><br>
 * Implementa los siguientes algoritmos:
 * <ul>
 *  <li> ENN: Regla de edición del vecino más cercano.
 *  <li> ENNTh: ENN Threshold.
 * </ul>  
 * </p>
 * <p>
 * <b>Funcionalidad</b><br>
 * Utiliza la biblioteca de algoritmos de selección de instancias realizada para el proyecto de final
 * de carrera en la Universidad de Burgos. Tutelado por: César García Osorio y Juan José Rodríguez Díez.
 * </p>
 * 
 * @author Álvar Arnaiz González
 * @version 1.8
 */
public class InstanceSelection extends Filter implements SupervisedFilter, OptionHandler, InstanceSelectionFilterIF {

	/**
	 * Serial UID.
	 */
	private static final long serialVersionUID = 8190462201379437586L;

	/**
	 * Algoritmo de selección de instancias a utilizar.
	 */
	protected main.core.algorithm.Algorithm mAlgorithm;
	
	/**
	 * Tiempo de CPU utilizado en el filtrado.
	 */
	protected long mCPUTimeElapsed;
	
	/**
	 * Tiempo utilizado en el filtrado.
	 */
	protected long mUserTimeElapsed;
	
	/**
	 * Número de vecinos cercanos a utilizar.
	 */
	protected int mNearestNeighbourNum = 1;
	
	/**
	 * Valor umbral Mu.
	 */
	protected double mMu = 0.7;
	
	/**
	 * Algoritmo ENN.
	 */
	public static final int TYPE_ENN = 0;
	
	/**
	 * Algoritmo ENNTh.
	 */
	public static final int TYPE_ENNTH = 1;
	
	/**
	 * Algoritmos implementados.
	 */
	public static final Tag[] TAGS_TYPE = {new Tag (TYPE_ENN, "ENN Filter"),
	                                       new Tag (TYPE_ENNTH, "ENNTh Algorithm")};

	
	/**
	 * Algoritmo por defecto.
	 */
	protected int mType = TYPE_ENN;
	
	/**
	 * Constructor por defecto.
	 */
	public InstanceSelection () {
		super();
	} // InstanceSelection
	
	/**
	 * Devuelve el número de vecinos a utilizar.
	 * 
	 * @return Número de vecinos cercanos a utilizar.
	 */
	public int getNumOfNearestNeighbour () {
		
		return mNearestNeighbourNum;
	} // getNumOfNearestNeighbour
	
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
	 * Establece el valor de Mu.
	 * 
	 * @param mu Umbral.
	 * @throws IllegalArgumentException
	 *             Es lanzada si el umbral está fuera del intervalo [0, 1]
	 */
	public void setMu (double mu) {
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
	 * Returns the tip text for this property.
	 * 
	 * @return Nearest neighbour number to use.
	 */
	public String numOfNearestNeighbourTipText () {
		
		return "Nearest neighbour number to used.";
	} // numOfNearestNeighbourTipText


	/**
	 * Returns the tip text for this property.
	 * 
	 * @return Threshold to use.
	 */
	public String muTipText () {
		
		return "Threshold (mu) to used (Only used by ENNth).";
	} // numMu

	/**
	 * Establece el tipo de algoritmo de selección de instancias a utilizar.
	 * 
	 * @param value Algoritmo seleccionado.
	 */
	public void setType (SelectedTag value) {
		if (value.getTags() == TAGS_TYPE)
			mType = value.getSelectedTag().getID();
	} // setType

	/**
	 * Deveuelve el algoritmo de selección de instancias a utilizar.
	 * 
	 * @return Algoritmo seleccionado.
	 */
	public SelectedTag getType () {
		
		return new SelectedTag(mType, TAGS_TYPE);
	} // getType
	
	/**
	 * Returns the tip text for this property.
	 * 
	 * @return Instance selection algorithm to use.
	 */
	public String typeTipText () {
		
		return "Instance selection algorithm to use.";
	} // typeTipText
	
	/**
	 * Devuelve las opciones del algoritmo.
	 * 
	 * @return Parámetros de algoritmo.
	 */
	public String[] getOptions () {
		Vector<String> result = new Vector<String>();
		
		result.add("-T");
		result.add("" + mType);
		
		result.add("-K");
		result.add("" + getNumOfNearestNeighbour());
		
		result.add("-M");
		result.add("" + getMu());
		
		return result.toArray(new String[result.size()]); 
	} // getOptions

	/**
	 * Lista los parámetros del algoritmo.
	 * 
	 * @return Parámetros del algoritmo.
	 */
	public Enumeration<Option> listOptions () {
		Vector<Option> newVector = new Vector<Option>();

		newVector.addElement(new Option("\tSpecifies the number of nearest neighbours\n" +
		                                "\t(default 1)", "K", 1, "-K <num>"));
		
		newVector.addElement(new Option("\tSet type of solver (default: 1)\n"+
		                                "\t\t 0 = ENN Filter\n"+
		                                "\t\t 1 = CNN Algorithm\n"+
		                                "\t\t 2 = RNN Algorithm\n"+
		                                "\t\t 3 = MSS Algorithm\n"+
		                                "\t\t 4 = ICF Algorithm\n"+
		                                "\t\t 5 = BSE Algorithm\n"+
		                                "\t\t 6 = Algoritmo DROP1\n" +
		                                "\t\t 7 = Algoritmo DROP2\n" +
		                                "\t\t 8 = Algoritmo DROP3\n" +
		                                "\t\t 9 = Algoritmo DROP4\n" +
		                                "\t\t 10 = Algoritmo DROP5\n" +
		                                "\t\t 11 = Algoritmo HMNE\n" +
		                                "\t\t 12 = Algoritmo HMNEI\n" +
		                                "\t\t 13 = Algoritmo CCIS\n" +
		                                "\t\t 14 = Algoritmo LSSm\n" +
		                                "\t\t 15 = Algoritmo LSBo\n" +
		                                "\t\t 16 = Algoritmo ENNTh\n"+
		                                "\t\t 17 = Algoritmo NCN\n",
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
		String tmpStr = Utils.getOption('T', options);
		String doubleStr = Utils.getOption('M', options);
		
		// Si el número de vecinos cercanos es distinto de 0 se asigna, sino se utilizará 1.
		if (numStr.length() != 0)
			setNumOfNearestNeighbour(Integer.parseInt(numStr));
		else
	    	setNumOfNearestNeighbour(1);
	    
		// Si el número de vecinos cercanos es distinto de 0 se asigna, sino se utilizará 0.7.
		if (doubleStr.length() != 0)
			setMu(Double.parseDouble(doubleStr));
		else
	    	setMu(0.7);
	    
		// Si el tipo de algoritmo es distinto de 0 se asigna, sino se utilizará del CNN.
	    if (tmpStr.length() != 0)
	    	setType(new SelectedTag(Integer.parseInt(tmpStr), TAGS_TYPE));
	    else
	    	setType(new SelectedTag(TYPE_ENN, TAGS_TYPE));
	} // setOptions

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
			if (mType == TYPE_ENN) {
				mAlgorithm = new ENNAlgorithm(inst);
				((ENNAlgorithm)mAlgorithm).setNumOfNearestNeighbour(mNearestNeighbourNum);
			} else if (mType == TYPE_ENNTH){
				mAlgorithm = new ENNThAlgorithm(inst);
				((ENNThAlgorithm)mAlgorithm).setNumOfNearestNeighbour(mNearestNeighbourNum);
				((ENNThAlgorithm)mAlgorithm).setMu(mMu);
			}
		}catch (NotEnoughInstancesException ex) {
			ex.printStackTrace();
			throw new IllegalStateException("The dataset has not enough instances");
		} catch (IllegalArgumentException ex) {
			throw new Exception("The neighbour number is wrong");
		} catch (Exception ex) {
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
	 * Returns the Capabilities of this filter.
	 * 
	 * @return the capabilities of this object
	 * @see Capabilities
	 */
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		
		// class
		result.disable(Capability.NUMERIC_CLASS);
		result.disable(Capability.DATE_CLASS);
		result.enable(Capability.NOMINAL_CLASS);

		return result;
	} // getCapabilities
	
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

} // InstanceSelection
