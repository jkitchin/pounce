//! A simplex vertex the never-regress gate declines can still end as an
//! accepted vertex from the active-set bridge.
//!
//! Both LPs are seeded random instances with rows and columns scaled across
//! four to six decades, the kind on which the simplex reaches an optimal
//! basis whose point or duals, read off its LU factor, carry more rounding
//! than the interior iterate. Each pins one of the two bridge starts:
//!
//! - 14 x 14 (`seed_126`): the bridge started on the simplex basis, pinned,
//!   recomputes point and duals from its own KKT solve and passes the gate.
//! - 16 x 20 (`seed_311`): the basis start yields nothing (its pinned KKT
//!   does not factor); the bridge from the interior iterate's active set
//!   reaches another optimal vertex that passes.
//!
//! The literals were generated, and the margins below measured, on macOS
//! arm64 only. Against the gate's threshold (interior KKT error + `1e-9`):
//! seed 126 declines the simplex vertex at 3.3x the threshold and accepts the
//! bridge vertex at 0.005x; seed 311 declines at 3.6x and accepts at 0.02x.
//! The tightest step is seed 311's basis start, a factorization verdict
//! rather than a gate one. Seeds were chosen for these margins; a seed within
//! about 2x of the threshold on either side was passed over.
//!
//! Neither is reachable if a declined simplex vertex ends crossover, which an
//! earlier draft of the declined-vertex fix did.

use pounce_convex::crossover::{CrossoverReport, clear_report, maybe_crossover, take_report};
use pounce_convex::{QpOptions, QpProblem, QpSolution, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// Crossover on the interior iterate of `prob`: the iterate, the result, the
/// record, and how many bridge solves (one backend each) it ran.
fn crossover(prob: &QpProblem) -> (QpSolution, QpSolution, CrossoverReport, u32) {
    let interior = solve_qp_ipm(prob, &QpOptions::default(), backend);
    let mut bridge_solves = 0_u32;
    let mut counting_backend = || {
        bridge_solves += 1;
        backend()
    };
    let on = QpOptions {
        crossover: true,
        ..QpOptions::default()
    };
    clear_report();
    let out = maybe_crossover(prob, interior.clone(), &on, &mut counting_backend);
    let report = take_report().expect("crossover ran");
    (interior, out, report, bridge_solves)
}

/// The record of a crossover whose simplex vertex was declined and whose
/// bridge vertex was accepted.
fn assert_bridge_rescued(interior: &QpSolution, out: &QpSolution, report: &CrossoverReport) {
    // Precondition: the simplex returned a vertex (pivot counts are recorded
    // only then). Had the gate accepted it, the engine would be `simplex`.
    // If this fails, the instance no longer declines the simplex vertex on
    // this build: regenerate a declined one rather than drop the test.
    assert!(
        report.pivots_push + report.pivots_phase1 + report.pivots_phase2 > 0,
        "precondition: the simplex returned no vertex: {report:?}"
    );
    assert_eq!(report.engine, "active-set", "{report:?}");
    assert!(
        report.accepted,
        "the bridge vertex was declined: {report:?}"
    );
    assert_ne!(
        out.x, interior.x,
        "the interior iterate was kept: {report:?}"
    );
}

#[test]
fn a_declined_simplex_vertex_is_rescued_from_its_basis() {
    let (interior, out, report, bridge_solves) = crossover(&seed_126());
    assert_bridge_rescued(&interior, &out, &report);
    assert_eq!(bridge_solves, 1, "rescued by the basis start: {report:?}");
}

#[test]
fn a_declined_simplex_vertex_is_rescued_from_the_interior_iterate() {
    let (interior, out, report, bridge_solves) = crossover(&seed_311());
    assert_bridge_rescued(&interior, &out, &report);
    assert_eq!(
        bridge_solves, 2,
        "the basis start, then the interior one: {report:?}"
    );
}

/// Seed 126 of the generator in the PR's probe (n, m, data below).
fn seed_126() -> QpProblem {
    QpProblem {
        // 126: n 14 m 14 k 9 spread 4
        n: 14,
        c: vec![
            -1141.7630226575127,
            897.6050907778819,
            -33851.885051238736,
            67177.99438134483,
            -507.2527576922356,
            267.98554587424815,
            -0.38755791069181994,
            1091.622766538005,
            -24326.69036306218,
            2509.6484965002765,
            -345.46798244559017,
            11927.293045029632,
            4413.181033705515,
            36998.15446167409,
        ],
        g: vec![
            Triplet::new(0, 0, 0.01573478955962065),
            Triplet::new(0, 1, 1.1188271204968103),
            Triplet::new(0, 4, 0.14321563838032755),
            Triplet::new(0, 5, 0.21377910317301185),
            Triplet::new(0, 8, 6.1578060134250485),
            Triplet::new(0, 10, -0.19941974862596723),
            Triplet::new(1, 3, -945.2447956987518),
            Triplet::new(1, 11, -122.45688893072989),
            Triplet::new(1, 13, -383.93256251035245),
            Triplet::new(2, 2, 1.3267128777610229),
            Triplet::new(2, 3, 12.239213934646168),
            Triplet::new(2, 7, 0.2164653249688224),
            Triplet::new(2, 12, 1.4504821528773244),
            Triplet::new(3, 1, -2.991833562121924),
            Triplet::new(3, 4, -34.17680641033597),
            Triplet::new(3, 6, -0.27891029717859944),
            Triplet::new(3, 7, -46.298735301362),
            Triplet::new(3, 8, 841.8383671388065),
            Triplet::new(3, 11, -236.09004817366394),
            Triplet::new(3, 13, -33.81882949486997),
            Triplet::new(4, 0, 1.6937727319141769),
            Triplet::new(4, 1, -6.793506313961043),
            Triplet::new(4, 2, 86.12189336144102),
            Triplet::new(4, 4, 4.377512024634354),
            Triplet::new(4, 5, 1.1900412401017821),
            Triplet::new(4, 6, 0.004434738553533522),
            Triplet::new(4, 13, 1.1396331065246572),
            Triplet::new(5, 0, 22.667752748476314),
            Triplet::new(5, 2, 1271.7497953495363),
            Triplet::new(5, 4, 41.630881575173895),
            Triplet::new(5, 7, -200.69237236862332),
            Triplet::new(5, 13, 20.091436388039398),
            Triplet::new(6, 0, 7.04020119107459),
            Triplet::new(6, 2, 166.74952616339868),
            Triplet::new(6, 3, 244.1542531079479),
            Triplet::new(6, 5, -2.5525571245333896),
            Triplet::new(6, 6, 0.007814246298299162),
            Triplet::new(6, 12, -36.17266357771033),
            Triplet::new(7, 1, -0.06410838982369527),
            Triplet::new(7, 2, -7.266131078325452),
            Triplet::new(7, 7, -0.9275360963696649),
            Triplet::new(7, 11, 5.750717924809506),
            Triplet::new(8, 1, -1.9382488420417554),
            Triplet::new(8, 3, -53.41888601772939),
            Triplet::new(8, 5, -0.7237956833093866),
            Triplet::new(8, 8, 163.93530556486354),
            Triplet::new(8, 9, -19.843395721783086),
            Triplet::new(8, 10, 2.8435202764743415),
            Triplet::new(9, 1, -34.05932501546717),
            Triplet::new(9, 5, 16.050615579861674),
            Triplet::new(9, 7, 9.104328674870505),
            Triplet::new(9, 9, 219.91741693405342),
            Triplet::new(10, 0, 0.38338689300002465),
            Triplet::new(10, 2, -2.68355518013743),
            Triplet::new(10, 4, -0.8399301212836341),
            Triplet::new(10, 5, 0.23981118037166135),
            Triplet::new(10, 8, -18.734036689073363),
            Triplet::new(10, 9, -2.0910186183622863),
            Triplet::new(10, 12, -13.71155329530806),
            Triplet::new(11, 0, 0.8842894426844462),
            Triplet::new(11, 4, -1.359152006585831),
            Triplet::new(11, 5, 0.6974300362250512),
            Triplet::new(11, 6, 0.0020459260096776064),
            Triplet::new(11, 10, 1.1800749799046282),
            Triplet::new(11, 11, -63.24036173570774),
            Triplet::new(12, 0, -0.5597493899382875),
            Triplet::new(12, 1, 12.382460005663772),
            Triplet::new(12, 3, -154.20604959923),
            Triplet::new(12, 4, -7.320963015289766),
            Triplet::new(12, 5, -2.3279122859956014),
            Triplet::new(12, 6, 0.03660836335417818),
            Triplet::new(12, 10, 0.8028036697083152),
            Triplet::new(12, 11, 53.98960466846351),
            Triplet::new(13, 1, 6.19685128068406),
            Triplet::new(13, 2, 21.762935964633943),
            Triplet::new(13, 4, 6.535617815545244),
            Triplet::new(13, 7, -24.435694573906094),
            Triplet::new(13, 9, -27.054577285700088),
            Triplet::new(13, 11, 81.67347773011282),
            Triplet::new(13, 12, -291.03509109724206),
            Triplet::new(13, 13, -46.26686848089212),
        ],
        h: vec![
            -3.1268717652273184,
            420.8586436039275,
            -10.992752357203313,
            -348.5453064215724,
            -0.4101812859954199,
            100.51383505719178,
            -185.50300555390555,
            4.1205378369716925,
            9.069113938790984,
            -102.51368193787123,
            14.444770830772296,
            -22.60941762680673,
            148.22282541739392,
            201.96625829639467,
        ],
        lb: vec![
            -0.0796850460811176,
            -4.251443845246241,
            -1.3374971364606412,
            -0.8088121569279463,
            0.10349416036867432,
            -3.6685066281517704,
            -0.747628678825486,
            -5.654275602838986,
            -4.7630777144616285,
            -0.5190546456330878,
            -4.0117560190765555,
            -2.0812174402177037,
            -5.425578162627476,
            0.746216632624104,
        ],
        ub: vec![
            0.39483715010281206,
            4.004631668392425,
            -0.07976106092934732,
            0.7985246317983279,
            3.392652527767467,
            -0.9902827614296699,
            2.676008932256826,
            2.3565732516514384,
            0.8579768618126744,
            2.0555609972551636,
            5.4801358467666565,
            3.4894987526850523,
            1.0473836051102023,
            2.9963533058724323,
        ],
        p_lower: vec![],
        a: vec![],
        b: vec![],
    }
}

/// Seed 311 of the generator in the PR's probe (n, m, data below).
fn seed_311() -> QpProblem {
    QpProblem {
        // 311: n 16 m 20 k 14 spread 6
        n: 16,
        c: vec![
            66430.7516021312,
            -53.196720447240985,
            -76.03734943807905,
            -0.39761666260409795,
            18324.325976456934,
            -3470.5696241414075,
            -8319.457027410477,
            542.5467692378396,
            -1185.406291203933,
            -4.135551408122126,
            -8.443381158078147,
            7585.264004445614,
            -282.1746470353398,
            0.06254698215267863,
            -0.18240491764174244,
            -1552329.4450522622,
        ],
        g: vec![
            Triplet::new(0, 0, 11.032304178608173),
            Triplet::new(0, 1, 0.00553412927867097),
            Triplet::new(0, 4, -3.8139652595463263),
            Triplet::new(0, 5, 1.2896240260304508),
            Triplet::new(0, 7, 0.03671655295868127),
            Triplet::new(0, 10, 0.0060822878425120805),
            Triplet::new(1, 0, -46.544765171671585),
            Triplet::new(1, 1, 0.04514836669856301),
            Triplet::new(1, 11, -1.74507689600931),
            Triplet::new(1, 12, 0.11272095630203499),
            Triplet::new(1, 14, 0.006029136532010009),
            Triplet::new(2, 0, -9.290618319156174),
            Triplet::new(2, 1, 0.06478983721857227),
            Triplet::new(2, 3, 0.05484975914037633),
            Triplet::new(2, 5, -0.8708398111845136),
            Triplet::new(2, 6, -0.6882959590268103),
            Triplet::new(2, 7, 0.02347541032484671),
            Triplet::new(2, 8, -32.852231506287936),
            Triplet::new(2, 9, 0.5161855000764972),
            Triplet::new(2, 10, 0.0006340683476023514),
            Triplet::new(2, 11, 0.9482247007896271),
            Triplet::new(2, 14, 0.01522049663231292),
            Triplet::new(3, 1, -0.007879525694197304),
            Triplet::new(3, 2, -0.003322460638668883),
            Triplet::new(3, 3, -0.0009220551317275127),
            Triplet::new(3, 5, 0.058887499417286285),
            Triplet::new(3, 7, -0.011113050291233047),
            Triplet::new(3, 12, -0.005381754320210653),
            Triplet::new(3, 15, 314.2086878171553),
            Triplet::new(4, 2, -0.0025861058141728724),
            Triplet::new(4, 3, 0.00046765679954102377),
            Triplet::new(4, 4, -0.07633012590966366),
            Triplet::new(4, 6, 0.030803297796634176),
            Triplet::new(4, 9, 0.029264251926211737),
            Triplet::new(4, 11, 0.04032192779038181),
            Triplet::new(4, 12, -0.0019548110380468693),
            Triplet::new(4, 13, 0.00014750169389553768),
            Triplet::new(4, 15, -3.254565844703117),
            Triplet::new(5, 0, 293.8197364006822),
            Triplet::new(5, 2, 1.3175512163413883),
            Triplet::new(5, 4, -481.50756652462917),
            Triplet::new(5, 10, 0.13942721461480995),
            Triplet::new(5, 11, 66.05755448204796),
            Triplet::new(5, 12, 0.2513749608166049),
            Triplet::new(6, 0, -0.10161966020550277),
            Triplet::new(6, 1, 0.0004339701150667814),
            Triplet::new(6, 3, -8.888516533941538e-05),
            Triplet::new(6, 4, -0.004767646160481587),
            Triplet::new(6, 7, 9.926590635598368e-05),
            Triplet::new(6, 8, 0.04899774054799681),
            Triplet::new(6, 10, -2.9817940367151117e-06),
            Triplet::new(6, 13, -1.1055206811586856e-05),
            Triplet::new(6, 14, -0.00015717161576055182),
            Triplet::new(6, 15, 3.1730785518032154),
            Triplet::new(7, 0, -957.8615815386335),
            Triplet::new(7, 5, 70.77719449735619),
            Triplet::new(7, 6, 170.25393853118962),
            Triplet::new(7, 7, -10.770317913858081),
            Triplet::new(7, 11, -207.95555176931674),
            Triplet::new(7, 12, 5.024817205276958),
            Triplet::new(7, 15, 31819.74205229725),
            Triplet::new(8, 0, 0.004988175317265762),
            Triplet::new(8, 2, -2.2758082613321492e-05),
            Triplet::new(8, 3, -2.413441789878278e-05),
            Triplet::new(8, 4, -5.399045039268769e-05),
            Triplet::new(8, 7, -1.9508511062043027e-05),
            Triplet::new(8, 9, 9.19630252592998e-05),
            Triplet::new(8, 11, -0.0004118607003139959),
            Triplet::new(8, 14, -6.6927879792633885e-06),
            Triplet::new(8, 15, 1.443043091317962),
            Triplet::new(9, 0, -0.06636387046381413),
            Triplet::new(9, 4, -0.03439935963089191),
            Triplet::new(9, 7, 4.5234602083946496e-05),
            Triplet::new(9, 8, -0.1846573368057831),
            Triplet::new(9, 9, 0.011115002368852834),
            Triplet::new(9, 10, -2.0892951033567296e-05),
            Triplet::new(9, 11, 0.003908433930670832),
            Triplet::new(9, 13, -7.437399710173313e-05),
            Triplet::new(10, 2, -0.046286634112753695),
            Triplet::new(10, 3, 0.012704006608033428),
            Triplet::new(10, 4, -2.832634103887914),
            Triplet::new(10, 6, -0.10218789959816445),
            Triplet::new(10, 8, 55.92727441517384),
            Triplet::new(10, 13, -0.0029401716903781866),
            Triplet::new(11, 1, -0.00013621401751253036),
            Triplet::new(11, 2, 0.0002584889654924503),
            Triplet::new(11, 6, 0.004580895014096722),
            Triplet::new(11, 8, 0.07516085765121908),
            Triplet::new(11, 13, 1.3970403890496697e-05),
            Triplet::new(11, 15, 3.4257779924651666),
            Triplet::new(12, 0, -1994.6992901476985),
            Triplet::new(12, 1, 3.3204383348567554),
            Triplet::new(12, 2, 1.2919612025012055),
            Triplet::new(12, 4, 166.27552952221225),
            Triplet::new(12, 6, 0.6897381194142577),
            Triplet::new(12, 7, -1.0974748599898265),
            Triplet::new(12, 10, 0.14910774305896332),
            Triplet::new(12, 11, -15.770386245942719),
            Triplet::new(12, 12, 1.489593013806831),
            Triplet::new(13, 1, -0.006314686252322958),
            Triplet::new(13, 6, 0.017848463654579927),
            Triplet::new(13, 9, 0.06870474802447146),
            Triplet::new(13, 12, -0.006006476179571688),
            Triplet::new(13, 15, -210.5853533035346),
            Triplet::new(14, 0, -6.718188881813947),
            Triplet::new(14, 2, -0.0062369689560093575),
            Triplet::new(14, 3, -0.014381072616606035),
            Triplet::new(14, 8, 14.288144097762713),
            Triplet::new(14, 10, 0.0017803554041560458),
            Triplet::new(14, 11, -0.044846768933391803),
            Triplet::new(14, 12, -0.04199384393791478),
            Triplet::new(14, 14, 0.0030143342111418354),
            Triplet::new(14, 15, 907.1724846731016),
            Triplet::new(15, 0, -164.38709539830123),
            Triplet::new(15, 1, -0.8084878062043502),
            Triplet::new(15, 3, -0.17161229874067632),
            Triplet::new(15, 4, -7.694534832539055),
            Triplet::new(15, 7, 0.6909662362194161),
            Triplet::new(15, 8, -194.99034776483944),
            Triplet::new(15, 10, -0.03862740113945461),
            Triplet::new(15, 12, -2.860640837600563),
            Triplet::new(15, 13, 0.017502434477877603),
            Triplet::new(16, 1, 1.0824619549313297),
            Triplet::new(16, 2, 0.10891260585164722),
            Triplet::new(16, 3, -0.15678380071770953),
            Triplet::new(16, 4, -61.79531428774844),
            Triplet::new(16, 5, -5.3101477589305865),
            Triplet::new(16, 6, -5.666219916261241),
            Triplet::new(16, 7, 2.4662560558191395),
            Triplet::new(16, 12, -0.20541193562823779),
            Triplet::new(17, 0, -0.02502782476142257),
            Triplet::new(17, 1, -0.0003871786283272378),
            Triplet::new(17, 3, -3.8778893441476686e-05),
            Triplet::new(17, 8, 0.19283242884507687),
            Triplet::new(17, 9, -0.003182087216572609),
            Triplet::new(17, 10, 8.58038399437546e-06),
            Triplet::new(17, 12, 0.0003378372042419287),
            Triplet::new(17, 14, -6.940201970284693e-06),
            Triplet::new(18, 5, 0.0016641427097514598),
            Triplet::new(18, 6, 0.0007614453043723946),
            Triplet::new(18, 9, -0.006948881440158101),
            Triplet::new(18, 10, -1.265879288490378e-05),
            Triplet::new(18, 12, -0.00011136202713143413),
            Triplet::new(18, 13, -6.210683468064839e-07),
            Triplet::new(18, 15, -5.626586718638153),
            Triplet::new(19, 0, -0.7623467599651503),
            Triplet::new(19, 1, -0.0015749728756183437),
            Triplet::new(19, 2, 0.006744976834365275),
            Triplet::new(19, 3, -0.002058417480808842),
            Triplet::new(19, 6, -0.04112455288404384),
            Triplet::new(19, 7, 0.0016367913162547018),
            Triplet::new(19, 8, -2.2430288320652476),
            Triplet::new(19, 10, 0.00037750696384120184),
            Triplet::new(19, 12, -0.01083410417762111),
        ],
        h: vec![
            3.2069121417937727,
            4.335989700402527,
            -9.303201299930334,
            188.38062722902518,
            -1.8678000737066942,
            382.54260413927716,
            1.9313483462948853,
            19180.512920547226,
            0.8642989678993501,
            -0.017348738377916412,
            19.19822765949324,
            2.0755033950057165,
            70.25220847865425,
            -126.2023817914861,
            550.2233771132426,
            -32.35555642865445,
            49.967649084941186,
            8.72192206361535,
            -3.0146079014343132,
            7.940390870105588,
        ],
        lb: vec![
            -3.6632334609383044,
            -0.1919638356931248,
            -3.383612797290363,
            -1.3789271731374075,
            -4.309561221884585,
            -3.03185662673905,
            -0.767957240641629,
            -3.820624973346395,
            -1.2050174539473297,
            -3.2414727011580595,
            -0.9864675186446825,
            -0.54901874871565,
            -5.15048447710811,
            0.038615820815522506,
            -3.066315068156871,
            0.5993412419846393,
        ],
        ub: vec![
            2.917430517385,
            0.884800462763089,
            2.1930007837320034,
            3.1594915289275214,
            0.8704769656091906,
            2.251094587586273,
            -0.00772564048583213,
            -0.6116765587750084,
            3.8821790126980567,
            2.74879223916421,
            2.143028587100646,
            0.2980517047539588,
            0.6222387864250136,
            3.5060938313454937,
            0.01340470228086632,
            0.5993412419846393,
        ],
        p_lower: vec![],
        a: vec![],
        b: vec![],
    }
}
