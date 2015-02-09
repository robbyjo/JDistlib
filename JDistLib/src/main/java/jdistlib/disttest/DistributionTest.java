/*
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; version 3 of the License.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  http://www.r-project.org/Licenses/
 */
package jdistlib.disttest;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import jdistlib.Ansari;
import jdistlib.Binomial;
import jdistlib.ChiSquare;
import jdistlib.F;
import jdistlib.Normal;
import jdistlib.Poisson;
import jdistlib.SignRank;
import jdistlib.T;
import jdistlib.Wilcoxon;
import jdistlib.generic.GenericDistribution;
import jdistlib.math.approx.ApproximationFunction;
import static java.lang.Math.*;
import static jdistlib.math.Constants.M_1_SQRT_2PI;
import static jdistlib.math.Constants.M_PISQ_8;
import static jdistlib.math.MathFunctions.lchoose;
import static jdistlib.util.Utilities.*;
import static jdistlib.math.VectorMath.*;

/**
 * Comparing two distributions
 * @author Roby Joehanes
 *
 */
public class DistributionTest {
	// Needed for dip test
	private static final double[][] qDiptab = {
		{0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.132559548782689,0.157497369040235,0.187401878807559,0.20726978858736,0.223755804629222,0.231796258864192,0.237263743826779,0.241992892688593,0.244369839049632,0.245966625504691,0.247439597233262,0.248230659656638,0.248754269146416,0.249302039974259,0.249459652323225,0.24974836247845},
		{0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.108720593576329,0.121563798026414,0.134318918697053,0.147298798976252,0.161085025702604,0.176811998476076,0.186391796027944,0.19361253363045,0.196509139798845,0.198159967287576,0.199244279362433,0.199617527406166,0.199800941499028,0.199917081834271,0.199959029093075,0.199978395376082,0.199993151405815,0.199995525025673,0.199999835639211},
		{0.0833333333333333,0.0833333333333333,0.0833333333333333,0.0833333333333333,0.0833333333333333,0.0924514470941933,0.103913431059949,0.113885220640212,0.123071187137781,0.13186973390253,0.140564796497941,0.14941924112913,0.159137064572627,0.164769608513302,0.179176547392782,0.191862827995563,0.202101971042968,0.213015781111186,0.219518627282415,0.224339047394446,0.229449332154241,0.232714530449602,0.236548128358969,0.2390887911995,0.240103566436295,0.244672883617768},
		{0.0714285714285714,0.0714285714285714,0.0714285714285714,0.0725717816250742,0.0817315478539489,0.0940590181922527,0.103244490800871,0.110964599995697,0.117807846504335,0.124216086833531,0.130409013968317,0.136639642123068,0.144240669035124,0.159903395678336,0.175196553271223,0.184118659121501,0.191014396174306,0.198216795232182,0.202341010748261,0.205377566346832,0.208306562526874,0.209866047852379,0.210967576933451,0.212233348558702,0.212661038312506,0.21353618608817},
		{0.0625,0.0625,0.0656911994503283,0.0738651136071762,0.0820045917762512,0.0922700601131892,0.0996737189599363,0.105733531802737,0.111035129847705,0.115920055749988,0.120561479262465,0.125558759034845,0.141841067033899,0.153978303998561,0.16597856724751,0.172988528276759,0.179010413496374,0.186504388711178,0.19448404115794,0.200864297005026,0.208849997050229,0.212556040406219,0.217149174137299,0.221700076404503,0.225000835357532,0.233772919687683},
		{0.0555555555555556,0.0613018090298924,0.0658615858179315,0.0732651142535317,0.0803941629593475,0.0890432420913848,0.0950811420297928,0.0999380897811046,0.104153560075868,0.108007802361932,0.112512617124951,0.122915033480817,0.136412639387084,0.146603784954019,0.157084065653166,0.164164643657217,0.172821674582338,0.182555283567818,0.188658833121906,0.194089120768246,0.19915700809389,0.202881598436558,0.205979795735129,0.21054115498898,0.21180033095039,0.215379914317625},
		{0.05,0.0610132555623269,0.0651627333214016,0.0718321619656165,0.077966212182459,0.0852835359834564,0.0903204173707099,0.0943334983745117,0.0977817630384725,0.102180866696628,0.109960948142951,0.118844767211587,0.130462149644819,0.139611395137099,0.150961728882481,0.159684158858235,0.16719524735674,0.175419540856082,0.180611195797351,0.185286416050396,0.191203083905044,0.195805159339184,0.20029398089673,0.205651089646219,0.209682048785853,0.221530282182963},
		{0.0341378172277919,0.0546284208048975,0.0572191260231815,0.0610087367689692,0.0642657137330444,0.0692234107989591,0.0745462114365167,0.0792030878981762,0.083621033469191,0.0881198482202905,0.093124666680253,0.0996694393390689,0.110087496900906,0.118760769203664,0.128890475210055,0.13598356863636,0.142452483681277,0.150172816530742,0.155456133696328,0.160896499106958,0.166979407946248,0.17111793515551,0.175900505704432,0.181856676013166,0.185743454151004,0.192240563330562},
		{0.033718563622065,0.0474333740698401,0.0490891387627092,0.052719998201553,0.0567795509056742,0.0620134674468181,0.0660163872069048,0.0696506075066401,0.0733437740592714,0.0776460662880254,0.0824558407118372,0.088344627001737,0.0972346018122903,0.105130218270636,0.114309704281253,0.120624043335821,0.126552378036739,0.13360135382395,0.138569903791767,0.14336916123968,0.148940116394883,0.152832538183622,0.156010163618971,0.161319225839345,0.165568255916749,0.175834459522789},
		{0.0262674485075642,0.0395871890405749,0.0414574606741673,0.0444462614069956,0.0473998525042686,0.0516677370374349,0.0551037519001622,0.058265005347493,0.0614510857304343,0.0649164408053978,0.0689178762425442,0.0739249074078291,0.0814791379390127,0.0881689143126666,0.0960564383013644,0.101478558893837,0.10650487144103,0.112724636524262,0.117164140184417,0.121425859908987,0.126733051889401,0.131198578897542,0.133691739483444,0.137831637950694,0.141557509624351,0.163833046059817},
		{0.0218544781364545,0.0314400501999916,0.0329008160470834,0.0353023819040016,0.0377279973102482,0.0410699984399582,0.0437704598622665,0.0462925642671299,0.048851155289608,0.0516145897865757,0.0548121932066019,0.0588230482851366,0.0649136324046767,0.0702737877191269,0.0767095886079179,0.0811998415355918,0.0852854646662134,0.0904847827490294,0.0940930106666244,0.0974904344916743,0.102284204283997,0.104680624334611,0.107496694235039,0.11140887547015,0.113536607717411,0.117886716865312},
		{0.0164852597438403,0.022831985803043,0.0238917486442849,0.0256559537977579,0.0273987414570948,0.0298109370830153,0.0317771496530253,0.0336073821590387,0.0354621760592113,0.0374805844550272,0.0398046179116599,0.0427283846799166,0.047152783315718,0.0511279442868827,0.0558022052195208,0.059024132304226,0.0620425065165146,0.0658016011466099,0.0684479731118028,0.0709169443994193,0.0741183486081263,0.0762579402903838,0.0785735967934979,0.0813458356889133,0.0832963013755522,0.0926780423096737},
		{0.0111236388849688,0.0165017735429825,0.0172594157992489,0.0185259426032926,0.0197917612637521,0.0215233745778454,0.0229259769870428,0.024243848341112,0.025584358256487,0.0270252129816288,0.0286920262150517,0.0308006766341406,0.0339967814293504,0.0368418413878307,0.0402729850316397,0.0426864799777448,0.044958959158761,0.0477643873749449,0.0497198001867437,0.0516114611801451,0.0540543978864652,0.0558704526182638,0.0573877056330228,0.0593365901653878,0.0607646310473911,0.0705309107882395},
		{0.00755488597576196,0.0106403461127515,0.0111255573208294,0.0119353655328931,0.0127411306411808,0.0138524542751814,0.0147536004288476,0.0155963185751048,0.0164519238025286,0.017383057902553,0.0184503949887735,0.0198162679782071,0.0218781313182203,0.0237294742633411,0.025919578977657,0.0274518022761997,0.0288986369564301,0.0306813505050163,0.0320170996823189,0.0332452747332959,0.0348335698576168,0.0359832389317461,0.0369051995840645,0.0387221159256424,0.03993025905765,0.0431448163617178},
		{0.00541658127872122,0.00760286745300187,0.00794987834644799,0.0085216518343594,0.00909775605533253,0.00988924521014078,0.0105309297090482,0.0111322726797384,0.0117439009052552,0.012405033293814,0.0131684179320803,0.0141377942603047,0.0156148055023058,0.0169343970067564,0.018513067368104,0.0196080260483234,0.0206489568587364,0.0219285176765082,0.0228689168972669,0.023738710122235,0.0248334158891432,0.0256126573433596,0.0265491336936829,0.027578430100536,0.0284430733108,0.0313640941982108},
		{0.00390439997450557,0.00541664181796583,0.00566171386252323,0.00607120971135229,0.0064762535755248,0.00703573098590029,0.00749421254589299,0.00792087889601733,0.00835573724768006,0.00882439333812351,0.00936785820717061,0.01005604603884,0.0111019116837591,0.0120380990328341,0.0131721010552576,0.0139655122281969,0.0146889122204488,0.0156076779647454,0.0162685615996248,0.0168874937789415,0.0176505093388153,0.0181944265400504,0.0186226037818523,0.0193001796565433,0.0196241518040617,0.0213081254074584},
		{0.00245657785440433,0.00344809282233326,0.00360473943713036,0.00386326548010849,0.00412089506752692,0.00447640050137479,0.00476555693102276,0.00503704029750072,0.00531239247408213,0.00560929919359959,0.00595352728377949,0.00639092280563517,0.00705566126234625,0.0076506368153935,0.00836821687047215,0.00886357892854914,0.00934162787186159,0.00993218636324029,0.0103498795291629,0.0107780907076862,0.0113184316868283,0.0117329446468571,0.0119995948968375,0.0124410052027886,0.0129467396733128,0.014396063834027},
		{0.00174954269199566,0.00244595133885302,0.00255710802275612,0.00273990955227265,0.0029225480567908,0.00317374638422465,0.00338072258533527,0.00357243876535982,0.00376734715752209,0.00397885007249132,0.00422430013176233,0.00453437508148542,0.00500178808402368,0.00542372242836395,0.00592656681022859,0.00628034732880374,0.00661030641550873,0.00702254699967648,0.00731822628156458,0.0076065423418208,0.00795640367207482,0.0082270524584354,0.00852240989786251,0.00892863905540303,0.00913853933000213,0.00952234579566773},
		{0.00119458814106091,0.00173435346896287,0.00181194434584681,0.00194259470485893,0.00207173719623868,0.00224993202086955,0.00239520831473419,0.00253036792824665,0.00266863168718114,0.0028181999035216,0.00299137548142077,0.00321024899920135,0.00354362220314155,0.00384330190244679,0.00420258799378253,0.00445774902155711,0.00469461513212743,0.00499416069129168,0.00520917757743218,0.00540396235924372,0.00564540201704594,0.00580460792299214,0.00599774739593151,0.00633099254378114,0.00656987109386762,0.00685829448046227},
		{0.000852415648011777,0.00122883479310665,0.00128469304457018,0.00137617650525553,0.00146751502006323,0.00159376453672466,0.00169668445506151,0.00179253418337906,0.00189061261635977,0.00199645471886179,0.00211929748381704,0.00227457698703581,0.00250999080890397,0.00272375073486223,0.00298072958568387,0.00315942194040388,0.0033273652798148,0.00353988965698579,0.00369400045486625,0.00383345715372182,0.00400793469634696,0.00414892737222885,0.0042839159079761,0.00441870104432879,0.00450818604569179,0.00513477467565583},
		{0.000644400053256997,0.000916872204484283,0.000957932946765532,0.00102641863872347,0.00109495154218002,0.00118904090369415,0.00126575197699874,0.00133750966361506,0.00141049709228472,0.00148936709298802,0.00158027541945626,0.00169651643860074,0.00187306184725826,0.00203178401610555,0.00222356097506054,0.00235782814777627,0.00248343580127067,0.00264210826339498,0.0027524322157581,0.0028608570740143,0.00298695044508003,0.00309340092038059,0.00319932767198801,0.00332688234611187,0.00339316094477355,0.00376331697005859}};
	private static final double[] qDiptabPr = {0,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,0.995,0.998,0.999,0.9995,0.9998,0.9999,0.99995,0.99998,0.99999,1};
	private static final int[] qDiptabN = {4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000, 72000};

	/**
	 * Compute the Kolmogorov-Smirnov test to test between two distribution, two-sided, exact p-value.
	 * If there are ties, then p-values will be inexact!
	 * 
	 * @param X an array with length of nX
	 * @param Y an array with length of nY
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] kolmogorov_smirnov_test(double[] X, double[] Y) {
		return kolmogorov_smirnov_test(X, Y, TestKind.TWO_SIDED, true);
	}

	/**
	 * Compute the Kolmogorov-Smirnov test to test between two distribution, two-sided.
	 * 
	 * @param X an array with length of nX
	 * @param Y an array with length of nY
	 * @param isExact whether the p-value should be computed with the exact method or not (takes a long time). If there are ties, this option is ignored.
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] kolmogorov_smirnov_test(double[] X, double[] Y, boolean isExact) {
		return kolmogorov_smirnov_test(X, Y, TestKind.TWO_SIDED, isExact);
	}

	/**
	 * Compute the Kolmogorov-Smirnov test to test between two distribution, exact p-value.
	 * If there are ties, then p-values will be inexact!
	 * 
	 * @param X an array with length of nX
	 * @param Y an array with length of nY
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] kolmogorov_smirnov_test(double[] X, double[] Y, TestKind kind) {
		return kolmogorov_smirnov_test(X, Y, kind, true);
	}

	/**
	 * Compute the Kolmogorov-Smirnov test to test between two distribution.
	 * 
	 * @param X an array with length of nX
	 * @param Y an array with length of nY
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @param isExact whether the p-value should be computed with the exact method or not (takes a long time). If there are ties, this option is ignored.
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] kolmogorov_smirnov_test(double[] X, double[] Y, TestKind kind, boolean isExact) {
		int
			nX = X.length,
			nY = Y.length,
			n_total = nX + nY;
		Set<Double> set = new HashSet<Double>();
		double w[] = c(X, Y);
		int[] ow = order(w);
		double z[] = new double[n_total], cs = 0;
		for (int i = 0; i < n_total; i++) {
			cs += (ow[i]+1 <= nX ? 1.0 / nX : -1.0/nY);
			z[i] = cs;
			set.add(w[i]);
		}
		int new_n = set.size();
		// Has ties
		if (new_n < n_total) {
			double[] dz = new double[n_total], new_z = new double[new_n];
			System.arraycopy(w, 0, dz, 0, n_total);
			sort(dz);
			dz = diff(dz);
			for (int i = 0, j = 0; i < dz.length; i++) {
				if (dz[i] != 0)
					new_z[j++] = z[i];
			}
			new_z[new_n-1] = z[n_total-1];
			z = new_z;
		}
		double maxDiv = Double.NaN, pval = maxDiv;
		switch(kind) {
			case TWO_SIDED: maxDiv = max(vabs(z)); break;
			case LOWER: maxDiv = -min(z); break;
			case GREATER: maxDiv = max(z); break;
		}

		// Has ties
		if (new_n < n_total || !isExact) {
			double n = nX * (nY * 1.0 / (nX + nY));
			if (kind != TestKind.TWO_SIDED) {
				pval = exp(-2 * n * maxDiv * maxDiv);
			} else {
				double tol = 1e-10;
				double d = sqrt(n) * maxDiv;
				if (d < 1) {
					int k_max = (int) sqrt(2 - log(tol));
					double
						z_star = -M_PISQ_8 / (d * d),
						w_star = log(d),
						s = 0;
					for (int k = 1; k < k_max; k += 2)
						s += exp(k * k * z_star - w_star);
					pval = 1 - s / M_1_SQRT_2PI;
				} else {
					int k = 1;
					double
						z_star = -2 * d * d,
						s = -1,
						oldval = 0,
						newval = 1;
					while (abs(oldval - newval) > tol) {
						oldval = newval;
						newval += 2 * s * exp(z_star * k * k);
						s *= -1;
						k++;
					}
					pval = 1 - newval;
				}
			}
		} else {
			if (nX > nY) {
				int temp = nY;
				nY = nX;
				nX = temp;
			}

			double
				q = (0.5 + floor(maxDiv * nX * nY - 1e-7)) / (nX * nY),
				u[] = new double[nY + 1],
				dx = 1.0 / nX, dy = 1.0 / nY;

			for (int j = 0; j <= nY; j++)
				u[j] = (j * dy) > q ? 0: 1;
			for(int i = 1; i <= nX; i++) {
				double w_star = (double)(i) / ((double)(i + nY));
				u[0] = (i * dx) > q ? 0 : w_star * u[0];
				for(int j = 1; j <= nY; j++)
					u[j] = abs(i * dx - j * dy) > q ? 0 : w_star * u[j] + u[j - 1];
			}
			pval = 1 - u[nY];
		}
		return new double[] { maxDiv, pval };
	}

	/**
	 * Compute the Kolmogorov-Smirnov test to test between X and a known reference distribution, two-sided, exact p-value.
	 * If there are ties, then p-values will be inexact!
	 * 
	 * @param X an array with length of nX
	 * @param dist reference distribution
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] kolmogorov_smirnov_test(double[] X, GenericDistribution dist) {
		return kolmogorov_smirnov_test(X, dist, TestKind.TWO_SIDED, true);
	}

	/**
	 * Compute the Kolmogorov-Smirnov test to test between X and a known reference distribution, exact p-value.
	 * If there are ties, then p-values will be inexact!
	 * 
	 * @param X an array with length of nX
	 * @param dist reference distribution
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] kolmogorov_smirnov_test(double[] X, GenericDistribution dist, TestKind kind) {
		return kolmogorov_smirnov_test(X, dist, kind, true);
	}

	/**
	 * Compute the Kolmogorov-Smirnov test to test between X and a known reference distribution, two-sided.
	 * 
	 * @param X an array with length of nX
	 * @param dist reference distribution
	 * @param isExact whether the p-value should be computed with the exact method or not (takes a long time). If there are ties, this option is ignored.
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] kolmogorov_smirnov_test(double[] X, GenericDistribution dist, boolean isExact) {
		return kolmogorov_smirnov_test(X, dist, TestKind.TWO_SIDED, isExact);
	}

	/**
	 * Compute the Kolmogorov-Smirnov test to test between X and a known reference distribution.
	 * 
	 * @param X an array with length of nX
	 * @param dist reference distribution
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @param isExact whether the p-value should be computed with the exact method or not (takes a long time). If there are ties, this option is ignored.
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] kolmogorov_smirnov_test(double[] X, GenericDistribution dist, TestKind kind, boolean isExact) {
		int n = X.length;
		double[] sortedX = new double[n];
		System.arraycopy(X, 0, sortedX, 0, n);
		sort(sortedX);
		boolean hasTies = false;
		double maxX = Long.MIN_VALUE, minX = Long.MAX_VALUE;
		for (int i = 0; i < n; i++) {
			if (i > 0 && sortedX[i] == sortedX[i - 1])
				hasTies = true;
			double val = dist.cumulative(sortedX[i]) - (i * 1.0 / n);
			if (maxX < val)
				maxX = val;
			else if (minX > val)
				minX = val;
		}
		double maxDiv = Double.NaN, pval = Double.NaN;
		switch(kind) {
			case TWO_SIDED: maxDiv = max(maxX, 1.0/n - minX); break;
			case LOWER: maxDiv = maxX; break;
			case GREATER: maxDiv = 1.0/n - minX; break;
		}
		pval = hasTies || !isExact ? kolmogorov_smirnov_pvalue_inexact(maxDiv, n) : kolmogorov_smirnov_pvalue_exact(maxDiv, n);
		return new double[] { maxDiv, pval };
	}

	static final double kolmogorov_smirnov_statistic(double[] X, GenericDistribution dist, TestKind kind) {
		int n = X.length;
		double[] sortedX = new double[n];
		System.arraycopy(X, 0, sortedX, 0, n);
		sort(sortedX);
		double maxX = Long.MIN_VALUE, minX = Long.MAX_VALUE;
		for (int i = 0; i < n; i++) {
			double val = dist.cumulative(sortedX[i]) - (i * 1.0 / n);
			if (maxX < val)
				maxX = val;
			else if (minX > val)
				minX = val;
		}
		switch(kind) {
			case TWO_SIDED: return max(maxX, 1.0/n - minX);
			case LOWER: return maxX;
			case GREATER: return 1.0/n - minX;
		}
		return Double.NaN;
	}

	static final double kolmogorov_smirnov_pvalue_inexact(double d, int n) {
		double pval;
		if (d >= 1)
			pval = 0;
		else if (d <= 0)
			pval = 1;
		else {
			int max_j = (int) floor(n * (1 - d));
			double s = 0, invN = 1.0 / n;
			for (int j = 0; j <= max_j; j++)
				s += exp(lchoose(n, j) + (n - j) * log(1 - d - j * invN) + (j - 1) * log(d + j * invN));
			pval = d * s;
		}
		return pval;
	}

	static final double kolmogorov_smirnov_pvalue_exact(double d, int n) {
		int
			k = (int) (n * d) + 1,
			m = 2 * k - 1,
			mm = m * m;
		double
			h = k - n * d,
			H[] = new double[mm],
			Q[] = new double[mm];
		for (int i = 0; i < m; i++)
			for (int j = 0; j < m; j++)
				H[i * m + j] = i - j + 1 < 0 ? 0 : 1;

		for(int i = 0; i < m; i++) {
			H[i * m] -= pow(h, i + 1);
			H[(m - 1) * m + i] -= pow(h, (m - i));
		}
		H[(m - 1) * m] += ((2 * h - 1 > 0) ? pow(2 * h - 1, m) : 0);
		for (int i = 0; i < m; i++)
			for (int j = 0; j < m; j++)
				if(i - j + 1 > 0)
					for(int g = 1; g <= i - j + 1; g++)
						H[i * m + j] /= g;
		int eH = 0;
		double eQ = m_power(H, eH, Q, 0, m, n);
		double s = Q[(k - 1) * m + k - 1];
		for(int i = 1; i <= n; i++) {
			s = s * i / n;
			if(s < 1e-140) {
				s *= 1e140;
				eQ -= 140;
			}
		}
		s *= pow(10., eQ);
		return 1 - s;
	}

	/**
	 * Helper function for Kolmogorov-Smirnov
	 * @param A
	 * @param eA
	 * @param V
	 * @param eV
	 * @param m
	 * @param n
	 * @return result of power series
	 */
	private static final int m_power(double[] A, int eA, double[] V, int eV, int m, int n) {
		double[] B = new double[m * m];
		int eB;

		if(n == 1) {
			for (int i = 0; i < m * m; i++)
				V[i] = A[i];
			return eA;
		}
		eV = m_power(A, eA, V, eV, m, n / 2);
		m_multiply(V, V, B, m);
		eB = 2 * eV;
		if((n & 1) == 0) {
			for (int i = 0; i < m * m; i++)
				V[i] = B[i];
			eV = eB;
		} else {
			m_multiply(A, B, V, m);
			eV = eA + eB;
	    }
		int mdiv2 = m / 2;
		if(V[mdiv2 * m + mdiv2] > 1e140) {
			for (int i = 0; i < m * m; i++)
				V[i] = V[i] * 1e-140;
			eV += 140;
		}
		return eV;
	}

	/**
	 * Helper function for Kolmogorov-Smirnov
	 * @param A
	 * @param B
	 * @param C
	 * @param m multiplication of a vector-shaped matrix
	 */
	private static final void m_multiply(double[] A, double[] B, double[] C, int m) {
		for (int i = 0; i < m; i++)
			for (int j = 0; j < m; j++) {
				double s = 0.0;
				for (int k = 0; k < m; k++)
					s+= A[i * m + k] * B[k * m + j];
				C[i * m + j] = s;
			}
	}

	/**
	 * Return the two-sided test of Ansari-Bradley.
	 * @param x the original x
	 * @param y the original y
	 * @param force_exact Set to true if you want exact answer. The default behavior is that
	 * if there are ties or either the length of x or the length of y is at least 50.
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] ansari_bradley_test(double[] x, double[] y, boolean force_exact) {
		return ansari_bradley_test(x, y, force_exact, TestKind.TWO_SIDED);
	}

	/**
	 * Ansari-Bradley test.
	 * @param x the original x
	 * @param y the original y
	 * @param force_exact Set to true if you want exact answer. The default behavior is that
	 * if there are ties or either the length of x or the length of y is at least 50.
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] ansari_bradley_test(double[] x, double[] y, boolean force_exact, TestKind kind) {
		int nx = x.length, ny = y.length;
		double N = nx + ny;
		double[] r = rank(c(x, y));
		boolean has_ties = unique(r).length != r.length;
		boolean large_xy = nx >= 50 || ny >= 50;
		double h = 0;
		r = pmin(r, vmin(N + 1, r));
		for (int i = 0; i < nx; i++)
			h += r[i];
		if (force_exact || (!has_ties && !large_xy)) {
			switch(kind) {
				case TWO_SIDED:
					double limit = (nx + 1) * (nx + 1) / 4 + (nx * ny / 2) / 2.0;
					double p = h > limit ? 1 - Ansari.cumulative((int) h - 1, nx, ny)
						: Ansari.cumulative((int) h, nx, ny);
					p = min(2 * p, 1);
					return new double[] {h, p};
				case LOWER:
					p = Ansari.cumulative((int) (h - 1), nx, ny, false);
					return new double[] {h, p};
				case GREATER:
					p = Ansari.cumulative((int) h, nx, ny, true);
					return new double[] {h, p};
			}
			throw new RuntimeException(); // Should never happen
		}
		// Has ties or large xy: Inexact match
		sort(r);
		double[][] rle = rle(r);
		double denom = 16 * N * (N - 1), Np1Sq = N + 1, Np2 = (N+2);
		Np1Sq *= Np1Sq;
		double sigma = 16 * sum(vtimes(vtimes(rle[0], rle[0]), rle[1]));
		sigma = N % 2 == 0 ? sqrt(nx * ny * (sigma - N*Np2*Np2) / denom)
			: sqrt(nx * ny * (sigma*N - Np1Sq*Np1Sq) / (denom * N));
		double z = N % 2 == 0 ? h - nx * Np2 / 4 : h - nx * Np1Sq / (4 * N);
		double p = Normal.cumulative(z/sigma, 0, 1, true, false);
		p = 2 * min(p, 1-p);
		return new double[] {h, p};
	}

	/**
	 * Performs Mood's two-sample test for a difference in scale parameters. Two-sided test. 
	 * @param x
	 * @param y
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] mood_test(double[] x, double[] y) {
		return mood_test(x, y, TestKind.TWO_SIDED);
	}
	/**
	 * Performs Mood's two-sample test for a difference in scale parameters. 
	 * @param x
	 * @param y
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] mood_test(double[] x, double[] y, TestKind kind) {
		int nx = x.length, ny = y.length;
		double N = nx + ny;
		if (N < 3)
			throw new RuntimeException("Not enough observations");
		double E = nx * (N * N - 1.) / 12.;
		double v = (1./180.) * nx * ny * (N + 1) * (N + 2) * (N - 2);
		double[] z = c(x, y);
		boolean has_ties = unique(z).length != z.length;
		double T = 0;
		double con = (N + 1.0)/ 2.;
		if (!has_ties) {
			double[] r = rank(z);
			for (int i = 0; i < nx; i++) {
				double val = r[i] - con;
				T += val * val;
			}
		} else {
			double[] u = unique(z);
			sort(u);
			int[] a = tabulate(match(x, u), u.length);
			int[] t = tabulate(match(z, u), u.length);
			double[] p = vmin(colon(1., z.length), con);
			p = cumsum(vsq(p));
			double sum = 0;
			for (int i = 0; i < u.length; i++) {
				double ti = t[i], NmtiSq = N - t[i], tsq = ti * ti;
				NmtiSq *= NmtiSq;
				sum += ti * (tsq - 1) * (tsq - 4 + 15 * NmtiSq);
			}
			v = v - (nx * ny) / (180. * N * (N - 1)) * sum;
			double[] temp = diff(c(new double[] {0.}, index_min1(p, cumsum(t))));
			for (int i = 0; i < t.length; i++)
				T += a[i] * temp[i] / t[i];
		}
		double zz = (T - E) / sqrt(v);
		double p = Normal.cumulative(zz, 0, 1, kind != TestKind.GREATER, false);
		return new double[] {zz, kind == TestKind.TWO_SIDED ? 2 * min(p, 1 - p) : p};
	}

	/**
	 * Performs an F test to compare the variances of two samples from normal populations. Ratio is set to 1.0. 
	 * @param x
	 * @param y
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] var_test(double[] x, double[] y, TestKind kind) {
		return var_test(x, y, 1, kind);
	}

	/**
	 * Performs an F test to compare the variances of two samples from normal populations. 
	 * @param x
	 * @param y
	 * @param ratio the hypothesized ratio of the population variances of x and y.
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] var_test(double[] x, double[] y, double ratio, TestKind kind) {
		double stat = (var(x) / var(y)) / ratio;
		double p = Double.NaN;
		switch (kind) {
			case TWO_SIDED:
				p = F.cumulative(stat, x.length - 1, y.length - 1, true, false);
				p = 2 * min(p, 1 - p);
				break;
			case GREATER:
				p = F.cumulative(stat, x.length - 1, y.length - 1, false, false);
				break;
			case LOWER:
				p = F.cumulative(stat, x.length - 1, y.length - 1, true, false);
				break;
		}
		return new double[] {stat, p};
	}

	/**
	 * One-sample Wilcoxon test. Test whether the vector of x is != mu
	 * @param x
	 * @param mu
	 * @param correction set to true if continuity correction is desired. Only matters
	 * if x has zeroes or ties
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] wilcoxon_test(double[] x, double mu, boolean correction, TestKind kind) {
		boolean has_zeroes = false;
		int n_nonzero = 0;
		Set<Double> seen_set = new HashSet<Double>();
		for (int i = 0; i < x.length; i++) {
			if (x[i] == mu) {
				has_zeroes = true;
			} else {
				n_nonzero++;
				seen_set.add(x[i]);
			}
		}
		double[] new_x = new double[n_nonzero];
		for (int i = 0, j = 0; i < x.length; i++)
			if (x[i] != mu)
				new_x[j++] = x[i] - mu;
		x = new_x;
		boolean has_ties = seen_set.size() != x.length;
		double[] r = rank(vabs(x));
		int n = r.length;
		double limit = n * (n + 1) / 4.0, v = 0, p = Double.NaN;
		for (int i = 0; i < n; i++)
			v += r[i];
		if (!has_ties && !has_zeroes) {
			SignRank sr = new SignRank(n);
			switch (kind) {
				case TWO_SIDED:
					p = v > limit ? sr.cumulative(v - 1, false, false) : sr.cumulative(v);
					p = min(2*p, 1);
					break;
				case GREATER:
					p = sr.cumulative(v - 1, false, false);
					break;
				case LOWER:
					p = sr.cumulative(v);
					break;
			}
		} else {
			double sigma = 0;
			for (int nties : table(r).values())
				sigma += (nties * nties * nties - nties);
			sigma = sqrt(limit * (2 * n + 1) / 6 - sigma / 48);
			v = v - limit;
			double cor = 0;
			if (correction) {
				switch (kind) {
					case TWO_SIDED:
						cor = signum(v) * 0.5; break;
					case GREATER:
						cor = 0.5; break;
					case LOWER:
						cor = -0.5; break;
				}
			}
			v = (v - cor) / sigma;
			switch (kind) {
				case TWO_SIDED:
					p = 2 * min (Normal.cumulative_standard(v),
						Normal.cumulative(v, 0, 1, false, false));
					break;
				case GREATER:
					p = Normal.cumulative(v, 0, 1, false, false); break;
				case LOWER:
					p = Normal.cumulative_standard(v); break;
			}
		}
		return new double[] {v, p};
	}

	/**
	 * Mann-Whitney-U test
	 * @param x
	 * @param y
	 * @param mu
	 * @param correction set to true if continuity correction is desired. Only matters
	 * then there are ties
	 * @param paired set to true for paired test (which reduces to Wilcoxon test)
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] mann_whitney_u_test(double[] x, double[] y, double mu, boolean correction,
		boolean paired, TestKind kind) {
		int nx = x.length, ny = y.length, n = nx + ny;
		if (paired)
			return wilcoxon_test(vmin(x, y), mu, correction, kind);
		double[] r = mu == 0 ? rank(c(x, y)) : rank(c(vmin(x, mu), y));
		double w = -nx * (nx + 1) / 2, p = Double.NaN, limit = nx * ny / 2.0;
		for (int i = 0; i < nx; i++)
			w += r[i];
		boolean has_ties = unique(r).length != r.length;
		if (!has_ties) {
			Wilcoxon wilcox = new Wilcoxon(nx, ny);
			switch (kind) {
				case TWO_SIDED:
					p = w > limit ? wilcox.cumulative(w - 1, false, false) : wilcox.cumulative(w);
					p = min(2*p, 1);
					break;
				case GREATER:
					p = wilcox.cumulative(w - 1, false, false);
					break;
				case LOWER:
					p = wilcox.cumulative(w);
					break;
			}
		} else {
			double sigma = 0;
			for (int nties : table(r).values())
				sigma += (nties * nties * nties - nties);
			sigma = sqrt((limit / 6) * ((n + 1) - sigma / (n * (n + 1))));
			w = w - limit;
			double cor = 0;
			if (correction) {
				switch (kind) {
					case TWO_SIDED:
						cor = signum(w) * 0.5; break;
					case GREATER:
						cor = 0.5; break;
					case LOWER:
						cor = -0.5; break;
				}
			}
			w = (w - cor) / sigma;
			switch (kind) {
				case TWO_SIDED:
					p = 2 * min (Normal.cumulative_standard(w),
						Normal.cumulative(w, 0, 1, false, false));
					break;
				case GREATER:
					p = Normal.cumulative(w, 0, 1, false, false); break;
				case LOWER:
					p = Normal.cumulative_standard(w); break;
			}
		}
		return new double[] {w, p};
	}

	/**
	 * One-sample t-test
	 * @param x
	 * @param mu
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] t_test(double[] x, double mu, TestKind kind) {
		int nx = x.length;
		double stderr = sqrt(var(x)/nx);
		nx--;
		double t = (mean(x) - mu) / stderr, p = Double.NaN;
		switch (kind) {
			case TWO_SIDED: p = 2 * T.cumulative(-abs(t), nx, true, false); break;
			case GREATER: p = T.cumulative(t, nx, false, false); break;
			case LOWER: p = T.cumulative(t, nx, true, false); break;
		}
		return new double[] {t, p};
	}

	/**
	 * Paired t-test
	 * @param x
	 * @param y
	 * @param mu
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] t_test_paired(double[] x, double[] y, double mu, TestKind kind) {
		return t_test(vmin(x, y), mu, kind);
	}

	/**
	 * Two sample t-test
	 * @param x
	 * @param y
	 * @param mu
	 * @param pool_var set to true if the variance should be pooled. Only matters when paired == false
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] t_test(double[] x, double[] y, double mu, boolean pool_var, TestKind kind) {
		int nx = x.length, ny = y.length;
		double stderr, df;
		if (pool_var) {
			df = nx + ny - 2;
			stderr = 0;
			if (nx > 1) stderr += (nx - 1) * var(x);
			if (ny > 1) stderr += (ny - 1) * var(y);
			stderr = sqrt((stderr/df) * (1.0/nx + 1.0/ny));
		} else {
			double sx = var(x)/nx, sy = var(y)/ny;
			stderr = sx + sy;
			df = stderr*stderr / (sx*sx/(nx-1) + sy*sy/(ny-1));
			stderr = sqrt(stderr);
		}
		double t = (mean(x) - mean(y) - mu) / stderr, p = Double.NaN;
		switch (kind) {
			case TWO_SIDED: p = 2 * T.cumulative(-abs(t), df, true, false); break;
			case GREATER: p = T.cumulative(t, df, false, false); break;
			case LOWER: p = T.cumulative(t, df, true, false); break;
		}
		return new double[] {t, p};
	}

	/**
	 * Binomial test
	 * @param n_success The number of successes
	 * @param n The total number of trials
	 * @param p Expected probability
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] binomial_test(int n_success, int n, double p, TestKind kind) {
		switch (kind) {
			case TWO_SIDED:
				if (p == 0) {
					p = n_success == 0 ? 1 : 0;
				} else if (p == 1) {
					p = n_success == n ? 1 : 0;
				} else {
					double d = Binomial.density(n_success, n, p, false) * (1 + 1e-7),
						m = n * p;
					if (n_success == m) {
						p = 1;
					} else if (n_success < m){
						int y = 0;
						for (int i = (int) ceil(m); i <= n; i++)
							if (Binomial.density(i, n, p, false) <= d) y++;
						p = Binomial.cumulative(n_success, n, p, true, false) +
							Binomial.cumulative(n - y, n, p, false, false);
					} else {
						int y = 0, mlo = (int) floor(m);
						for (int i = 0; i <= mlo; i++)
							if (Binomial.density(i, n, p, false) <= d) y++;
						p = Binomial.cumulative(y - 1, n, p, true, false) +
							Binomial.cumulative(n_success - 1, n, p, false, false);
					}
				}
				break;
			case GREATER: p = Binomial.cumulative(n_success - 1, n, p, false, false); break;
			case LOWER: p = Binomial.cumulative(n_success, n, p, true, false); break;
		}
		return new double[] {n_success, p};
	}

	/**
	 * Bartlett's test
	 * @param x
	 * @param group an array of group indices. Observation in x that belongs in the same group must have the same index.
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] bartlett_test(double[] x, int[] group) {
		int n = x.length;
		if (n != group.length)
			throw new RuntimeException();
		Map<Integer, List<Double>> map = new HashMap<Integer, List<Double>>();
		for (int i = 0; i < n; i++) {
			List<Double> ll = map.get(group[i]);
			if (ll == null) {
				ll = new ArrayList<Double>();
				map.put(group[i], ll);
			}
			ll.add(x[i]);
		}
		int[] unique_group = to_int_array(map.keySet());
		int k = unique_group.length;
		double v_total = 0, sum_recip = 0, sum_n_vlog = 0;
		for (int i = 0; i < k; i++) {
			double[] dbl = to_double_array(map.get(unique_group[i]));
			double var_group = var(dbl);
			int ni = dbl.length - 1;
			v_total += ni * var_group / (n - k); 
			sum_recip += 1.0/ni;
			sum_n_vlog += ni * log(var_group);
		}
		double stat = (((n - k) * log(v_total) - sum_n_vlog) / (1 + (sum_recip - 1.0/(n-k)) / (3*(k-1))));
		double p = ChiSquare.cumulative(stat, k - 1, false, false);
		return new double[] {stat, p};
	}

	/**
	 * Fligner-Killeen test
	 * @param x
	 * @param group an array of group indices. Observation in x that belongs in the same group must have the same index.
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] fligner_test(double[] x, int[] group) {
		int n = x.length;
		if (n != group.length)
			throw new RuntimeException();
		Map<Integer, List<Double>> map = new HashMap<Integer, List<Double>>();
		for (int i = 0; i < n; i++) {
			List<Double> ll = map.get(group[i]);
			if (ll == null) {
				ll = new ArrayList<Double>();
				map.put(group[i], ll);
			}
			ll.add(x[i]);
		}
		int[] unique_group = to_int_array(map.keySet());
		int k = unique_group.length;
		int[] cumsum_n_group = new int[k], n_group = new int[k];
		double[] new_x = new double[n];
		for (int i = 0; i < k; i++) {
			double[] dbl = to_double_array(map.get(unique_group[i]));
			int ni = dbl.length;
			cumsum_n_group[i] = ni + (i > 0 ? cumsum_n_group[i-1] : 0);
			n_group[i] = ni;
			double med_group = median(dbl);
			for (int j = 0; j < ni; j++)
				dbl[j] -= med_group;
			System.arraycopy(dbl, 0, new_x, i == 0 ? 0 : cumsum_n_group[i-1], ni);
		}
		new_x = rank(vabs(new_x));
		for (int i = 0; i < new_x.length; i++)
			new_x[i] = Normal.quantile((1 + new_x[i] / (n + 1)) / 2.0, 0, 1, true, false);
		double stat = 0;
		for (int i = 0; i < k; i++) {
			int
				from = i == 0 ? 0 : cumsum_n_group[i - 1],
				ni = n_group[i],
				to = from + ni;
			double sum = 0;
			for (int j = from; j < to; j++)
				sum += new_x[j];
			stat += sum * sum / ni;
		}
		double ma = mean(new_x);
		stat = (stat - n * ma * ma) / var(new_x);
		double p = ChiSquare.cumulative(stat, k - 1, false, false);
		return new double[] {stat, p};
	}

	/**
	 * Kruskal-Wallis test
	 * @param x
	 * @param group an array of group indices. Observation in x that belongs in the same group must have the same index.
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] kruskal_wallis_test(double[] x, int[] group) {
		int n = x.length;
		if (n != group.length)
			throw new RuntimeException();
		Map<Integer, List<Double>> map = new HashMap<Integer, List<Double>>();
		for (int i = 0; i < n; i++) {
			List<Double> ll = map.get(group[i]);
			if (ll == null) {
				ll = new ArrayList<Double>();
				map.put(group[i], ll);
			}
			ll.add(x[i]);
		}
		int[] unique_group = to_int_array(map.keySet());
		int k = unique_group.length;
		int[] cumsum_n_group = new int[k], n_group = new int[k];
		double[] new_x = new double[n];
		for (int i = 0; i < k; i++) {
			double[] dbl = to_double_array(map.get(unique_group[i]));
			int ni = dbl.length;
			cumsum_n_group[i] = ni + (i > 0 ? cumsum_n_group[i-1] : 0);
			n_group[i] = ni;
			System.arraycopy(dbl, 0, new_x, i == 0 ? 0 : cumsum_n_group[i-1], ni);
		}
		double[] r = rank(new_x);
		double stat = 0;
		for (int i = 0; i < k; i++) {
			int
				from = i == 0 ? 0 : cumsum_n_group[i - 1],
				ni = n_group[i],
				to = from + ni;
			double sum = 0;
			for (int j = from; j < to; j++)
				sum += r[j];
			stat += sum * sum / ni;
		}
		double sigma = 0;
		for (int nties : table(r).values())
			sigma += (nties * nties * nties - nties);
		stat = ((12 * stat / (n * (n + 1)) - 3 * (n + 1)) / (1 - sigma / (n*n*n - n)));
		double p = ChiSquare.cumulative(stat, k - 1, false, false);
		return new double[] {stat, p};
	}

	/**
	 * Performs an exact test of a simple null hypothesis about the rate parameter in Poisson distribution
	 * @param num_events number of events.
	 * @param time time base for event count.
	 * @param rate hypothesized rate
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] poisson_test(int num_events, double time, double rate, TestKind kind) {
		if (time < 0 || rate < 0 || num_events < 0) throw new RuntimeException();
		double m = rate * time, p = Double.NaN;
		switch (kind) {
			case TWO_SIDED:
				if (m == 0) {
					p = num_events == 0 ? 1 : 0; break;
				}
				if (num_events == m) {
					p = 1; break;
				}
				double d = Poisson.density(num_events, m, false), fuzz = (1 + 1e-7) * d, y = 0, N;
				if (num_events < m) {
					N = (int) ceil(2 * m - num_events);
					while (Poisson.density(N, m, false) > d) N *= 2;
					for (int i = (int) ceil(m); i <= N; i++)
						if (Poisson.density(i, m, false) <= fuzz)
							y++;
					p = Poisson.cumulative(num_events, m, true, false) +
						Poisson.cumulative(N - y, m, false, false);
				} else {
					N = (int) floor(m);
					for (int i = 0; i < N; i++)
						if (Poisson.density(i, m, false) <= fuzz)
							y++;
					p = Poisson.cumulative(y - 1, m, true, false) +
						Poisson.cumulative(num_events - 1, m, false, false);
				}
				break;
			case GREATER: p = Poisson.cumulative(num_events-1, m, false, false); break;
			case LOWER: p = Poisson.cumulative(num_events, m, true, false); break;
		}
		return new double[] {num_events, p};
	}

	/**
	 * Comparison of Poisson rates
	 * @param num_events1 number of events for the treatment.
	 * @param num_events2 number of events for control.
	 * @param time1 time base for event count for treatment.
	 * @param time2 time base for event count for control.
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] poisson_test(int num_events1, int num_events2, double time1, double time2, double r, TestKind kind) {
		if (time1 < 0 || time2 < 0 || r < 0 || num_events1 < 0 || num_events2 < 0) throw new RuntimeException();
		return binomial_test(num_events1, num_events1+num_events2, r * time1 / (r * time1 + time2), kind);
	}

	/**
	 * Two-sample Cramer-Von Mises test
	 * @param X
	 * @param Y
	 * @return statistic
	 */
	static final double cramer_vonmises_statistic(double[] X, double[] Y) {
		int
			nX = X.length,
			nY = Y.length,
			nXY = nX * nY,
			nXPY = nX + nY;
		double[] rank = rank(c(X, Y)),
			rankX = rank(X),
			rankY = rank(Y);
		double sumX = 0, sumY = 0, val;
		for (int i = 0; i < nX; i++) {
			val = rank[i] - rankX[i];
			sumX += val * val;
		}
		for (int i = nX; i < nXPY; i++) {
			val = rank[i] - rankY[i - nX];
			sumY += val * val;
		}
		val = (nX * sumX + nY * sumY) / (nXY * nXPY) - (4*nXY - 1) / (6 * nXPY); // T statistic

		/*
		int gcd = jdistlib.math.MathFunctions.gcd(nX, nY);
		int nL = nX / gcd * nY, nP = nL / nX, nQ = nL / nY;
		double coef = (((1.0/nP) * (1.0/nQ)) / nXPY) / nXPY;
		sumX = sumY = 0;
		for (int i = 0; i < nXPY; i++) {
			sumX += rank[i] < nX ? nP : -nQ;
			sumY += sumX*sumX;
		}
		val = sumY * coef;
		//*/
		return val;
	}

	/**
	 * Perform Hartigan's dip test, assuming the minimum test statistics D is zero.
	 * @param x Can be of any order. If x is already sorted, use diptest_presorted to save some time.
	 * @return an array of four elements: The first is the test statistic, the second is the p-value, followed by indices for which there are a dip. If there is no dip, the indices will be set to -1.
	 */
	public static final double[] diptest(double[] x) {
		double[] x_ = new double[x.length];
		System.arraycopy(x, 0, x_, 0, x.length);
		sort(x_);
		return diptest_presorted(x_);
	}

	/**
	 * Perform Hartigan's dip test, assuming the minimum test statistics D is zero.
	 * @param x MUST BE SORTED in order to output the right result. This routine will NOT check for order!
	 * @return an array of four elements: The first is the test statistic, the second is the p-value, followed by indices for which there are a dip. If there is no dip, the indices will be set to -1.
	 */
	public static final double[] diptest_presorted(double[] x) {
		double dip = 0, dip_l, dip_u, dipnew;
		int n = x.length, low = 1, high = n, l_gcm = 0, l_lcm = 0;
		int mnj, mnmnj, mjk, mjmjk, ig, ih, iv, ix, i;
		int[] gcm = new int[n+1], lcm = new int[n+1], mn = new int[n+1], mj = new int[n+1];
		if (n < 2 || x[0] == x[n-1]) return new double[] {0, 1, -1, -1};
		//* Establish the indices  mn[0..n-1]  over which combination is necessary for the convex MINORANT (GCM) fit.
		mn[1] = 1;
		for (int j = 2; j <= n; ++j) {
			mn[j] = j - 1;
			while(true) {
				mnj = mn[j];
				mnmnj = mn[mnj];
				if (mnj == 1 || ( x[j-1]  - x[mnj-1]) * (mnj - mnmnj) < (x[mnj-1] - x[mnmnj-1]) * (j - mnj)) break;
				mn[j] = mnmnj;
			}
		}
		// Establish the indices   mj[0..n-1]  over which combination is necessary for the concave MAJORANT (LCM) fit.
		mj[n] = n;
		for (int k = n - 1; k >= 1; k--) {
			mj[k] = k + 1;
			while(true) {
				mjk = mj[k];
				mjmjk = mj[mjk];
				if (mjk == n || ( x[k-1]  - x[mjk-1]) * (mjk - mjmjk) < (x[mjk-1] - x[mjmjk-1]) * (k - mjk)) break;
				mj[k] = mjmjk;
			}
		}

		/* ----------------------- Start the cycling. ------------------------------- */
		//LOOP_Start:
		while (true) {

			/* Collect the change points for the GCM from HIGH to LOW. */
			gcm[1] = high;
			for(i = 1; gcm[i] > low; i++)
				gcm[i+1] = mn[gcm[i]];
			ig = l_gcm = i; // l_gcm == relevant_length(GCM)
			ix = ig - 1; //  ix, ig  are counters for the convex minorant.

			/* Collect the change points for the LCM from LOW to HIGH. */
			lcm[1] = low;
			for(i = 1; lcm[i] < high; i++)
				lcm[i+1] = mj[lcm[i]];
			ih = l_lcm = i; // l_lcm == relevant_length(LCM)
			iv = 2; //  iv, ih  are counters for the concave majorant.

			//	Find the largest distance greater than 'DIP' between the GCM and the LCM from LOW to HIGH.

			// FIXME: <Rconfig.h>  should provide LDOUBLE or something like it
			/* long */ double d = 0.;// <<-- see if this makes 32-bit/64-bit difference go..
			if (l_gcm != 2 || l_lcm != 2) {
				//if(*debug) Rprintf("  while(gcm[ix] != lcm[iv]) :%s", (*debug >= 2) ? "\n" : " ");
				do { /* gcm[ix] != lcm[iv]  (after first loop) */
					/* long */ double dx;
					int gcmix = gcm[ix], lcmiv = lcm[iv];
					if (gcmix > lcmiv) {
						// If the next point of either the GCM or LCM is from the LCM, calculate the distance here.
						int gcmi1 = gcm[ix + 1];
						dx = (lcmiv - gcmi1 + 1) -
								(/*(long double)*/ x[lcmiv-1] - x[gcmi1-1]) * (gcmix - gcmi1)/(x[gcmix-1] - x[gcmi1-1]);
						++iv;
						if (dx >= d) {
							d = dx;
							ig = ix + 1;
							ih = iv - 1;
							//if(*debug >= 2) Rprintf(" L(%d,%d)", ig,ih);
						}
					}
					else {
						// If the next point of either the GCM or LCM is from the GCM, calculate the distance here.
						int lcmiv1 = lcm[iv - 1];
						/* Fix by Yong Lu {symmetric to above!}; original Fortran: only ")" misplaced! :*/
						dx = (/*(long double)*/x[gcmix-1] - x[lcmiv1-1]) * (lcmiv - lcmiv1) / (x[lcmiv-1] - x[lcmiv1-1])- (gcmix - lcmiv1 - 1);
						--ix;
						if (dx >= d) {
							d = dx;
							ig = ix + 1;
							ih = iv;
							//if(*debug >= 2) Rprintf(" G(%d,%d)", ig,ih);
						}
					}
					if (ix < 1)	ix = 1;
					if (iv > l_lcm)	iv = l_lcm;
					//if(*debug) {
					//	if(*debug >= 2) Rprintf(" --> ix = %d, iv = %d\n", ix,iv);
					//	else Rprintf(".");
					//}
				} while (gcm[ix] != lcm[iv]);
				//if(*debug && *debug < 2) Rprintf("\n");
			}
			else { /* l_gcm or l_lcm == 2 */
				d = 0; // d = (*min_is_0) ? 0. : 1.;
				//if(*debug) Rprintf("  ** (l_lcm,l_gcm) = (%d,%d) ==> d := %g\n", l_lcm, l_gcm, (double)d);
			}

			if (d < dip) break; // goto L_END;

			// Calculate the DIPs for the current LOW and HIGH.
			//if(*debug) Rprintf("  calculating dip ..");

			//int j_best, j_l = -1, j_u = -1;

			/* The DIP for the convex minorant. */
			dip_l = 0.;
			for (int j = ig; j < l_gcm; ++j) {
				double max_t = 1.;
				int /*j_ = -1,*/ jb = gcm[j + 1], je = gcm[j];
				if (je - jb > 1 && x[je-1] != x[jb-1]) {
					double C = (je - jb) / (x[je-1] - x[jb-1]);
					for (int jj = jb; jj <= je; ++jj) {
						double t = (jj - jb + 1) - (x[jj-1] - x[jb-1]) * C;
						if (max_t < t) {
							max_t = t; //j_ = jj;
						}
					}
				}
				if (dip_l < max_t) {
					dip_l = max_t; //j_l = j_;
				}
			}

			/* The DIP for the concave majorant. */
			dip_u = 0.;
			for (int j = ih; j < l_lcm; ++j) {
				double max_t = 1.;
				int /*j_ = -1,*/ jb = lcm[j], je = lcm[j + 1];
				if (je - jb > 1 && x[je-1] != x[jb-1]) {
					double C = (je - jb) / (x[je-1] - x[jb-1]);
					for (int jj = jb; jj <= je; ++jj) {
						double t = (x[jj-1] - x[jb-1]) * C - (jj - jb - 1);
						if (max_t < t) {
							max_t = t; //j_ = jj;
						}
					}
				}
				if (dip_u < max_t) {
					dip_u = max_t; //j_u = j_;
				}
			}

			//if(*debug) Rprintf(" (dip_l, dip_u) = (%g, %g)", dip_l, dip_u);

			/* Determine the current maximum. */
			if(dip_u > dip_l) {
				dipnew = dip_u; //j_best = j_u;
			} else {
				dipnew = dip_l; //j_best = j_l;
			}
			if (dip < dipnew) {
				dip = dipnew;
				//if(*debug) Rprintf(" -> new larger dip %g (j_best = %d)\n", dipnew, j_best);
			}
			//else if(*debug) Rprintf("\n");

			/*--- The following if-clause is NECESSARY  (may loop infinitely otherwise)!
		      --- Martin Maechler, Statistics, ETH Zurich, July 30 1994 ---------- */
			if (low == gcm[ig] && high == lcm[ih]) {
				//if(*debug) Rprintf("No improvement in  low = %ld  nor  high = %ld --> END\n", low, high);
				break;
			} else {
				low  = gcm[ig];
				high = lcm[ih];	// goto LOOP_Start; /* Recycle */
			}
		}
		/*---------------------------------------------------------------------------*/

		//L_END:
		/* do this in the caller :
		 *   *xl = x[low];  *xu = x[high];
		 * rather return the (low, high) indices -- automagically via lo_hi[]  */
		dip /= (2*n);
		int nn = qDiptabN.length - 1, max_n = qDiptabN[nn], prn = qDiptab[0].length;
		double zz[] = new double[prn];
		if (n > max_n) {
			double sqn0 = sqrt(max_n);
			for (int j = 0; j < prn; j++)
				zz[j] = sqn0 * qDiptab[nn][j];
		} else {
			int i_n = nn;
			while(n < qDiptabN[i_n]) i_n--;
			int i2 = i_n + 1,
				n0 = qDiptabN[i_n],
				n1 = qDiptabN[i2];
			double f_n = (n - n0) * 1.0/(n1 - n0), sqn0 = sqrt(n0), sqn1 = sqrt(n1);
			for (int j = 0; j < prn; j++) {
				double y0 = sqn0 * qDiptab[i_n][j];
				zz[j] = y0 + f_n * (sqn1 * qDiptab[i2][j] - y0);
			}
		}
		double pval = 1 - ApproximationFunction.linear(sqrt(n) * dip, zz, qDiptabPr, 0, 1);
		return new double[] {dip, pval, low-1, high-1};
	}

	static final void kstest_example() {
		double x[] = { 1.16004168838821208886714, -1.05547634926559497081655, -1.32072420295666459466588,  0.23915046399456202363965,
			 0.12803724906074620548679,  0.05569133699728501252224, -0.81250026875197078890523, -0.25270560923205648284906,
			 -0.18064235632836450617944, -1.68851736711207101038212,  0.45201765941273730486927, -0.82239514187785234256012,
			 -1.28431746020543480213405, -1.11673394115548996197163,  0.35484674303354840629865,  0.69469717363334693160937,
			  0.80814838465816862811408, -0.21328821452795371227396, -1.27614688227021422228802,  0.70701146687565052939561,
			  0.50751733987764702238366,  0.03272845258879651664241,  0.35783108099085625397606,  0.44208900297507758292426,
			 -0.52069082447125036861024, -0.55763158717776695194601, -0.41633151696193471114071,  0.26769602419784399582880,
			 -0.74987234035877792237557, -0.41904535934255054963060,  0.89257906144225818145799, -0.07190597453573960295969,
			  0.73302279756488808448722, -0.10734082387514076728507,  0.69406629605031111562852, -0.15263137425121245382975,
			 -1.19674895163987238255743,  0.66757167860376609436202, -0.81494857018266397830075,  0.80040931005785931340313,
			  0.80754967242376574088070, -1.56916928352228968179816, -1.35167386137606548857093,  1.16318789818012624515120,
			  0.28936206000057662635072,  0.52290491081517120885991, -0.05762605570118027598081, -0.14176370966120629968366,
			 -0.37619927990739526757480, -1.06736558034971595887441};
		double y[] = { 0.40490121906623244285583, 0.24850796000100672245026, 0.10222866409458220005035, 0.62853105599060654640198,
				0.61955126281827688217163, 0.28701157541945576667786, 0.95149635546840727329254, 0.01087409863248467445374,
				0.63707210076972842216492, 0.61038276972249150276184, 0.09210657258518040180206, 0.64758135029114782810211,
				0.48832037649117410182953, 0.10307366680353879928589, 0.86465166741982102394104, 0.68774600140750408172607,
				0.76253556110896170139313, 0.52018100558780133724213, 0.61665696790441870689392, 0.77783631114289164543152,
				0.89877208555117249488831, 0.83583764336071908473969, 0.92252740426920354366302, 0.83699792949482798576355,
				0.35809992859140038490295, 0.59004115150310099124908, 0.60853263596072793006897, 0.13569264346733689308167,
				0.38345616171136498451233, 0.91171105671674013137817};
		double[] p = kolmogorov_smirnov_test(x, y);
		System.out.println(p[0]);
		System.out.println(p[1]);
		p = kolmogorov_smirnov_test(x, new Normal());
		System.out.println(p[0]);
		System.out.println(p[1]);
	}

	public static final void main(String[] args) {
		kstest_example();
		System.exit(0);
		double[] x = { -1.2315764307891696738295, 0.1076666048919862200828, -0.2507677102611699515577,	0.1865730243313593050836,
			0.7674721840239807635342, -0.1874640529241502207025, 0.1376975996921310230192, 0.3722658431557314684390,
			1.8257862598243677076937, -1.4691239378183402752853
		};
		double[] y = { 2.633833206002905935605, -1.041337574910569774289, -1.081121838223072728624, 2.702460192243479220053,
			1.626548966201278201282, 1.336642538096019183769, 1.075145021293279601338, 1.543056949670002397923,
			-0.085039987328253241472, 1.357930215887039437916
		};
		// Correct answer: T = 0.405, P-value: 0.07656584901166944845397
		System.out.println(cramer_vonmises_statistic(x, y));
	}
}
