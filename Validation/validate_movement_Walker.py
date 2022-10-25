#!/usr/bin/env python

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import time
import gym
import numpy as np

""" Minimun Environmnt just for walker"""
class MiniEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    collision_sensor: object

    def __init__(self):
        super(MiniEnv, self).__init__()

        # === Carla ===
        self.host = 'localhost'
        self.town = 'Town01'

        self.client = carla.Client(self.host, 2000)
        self.client.set_timeout(60.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.actor_list = []

        # === set correct map ===
        self.map = self.world.get_map()
        if not self.map.name.endswith(self.town):
            self.world = self.client.load_world(self.town)
            while not self.world.get_map().name.endswith(self.town):
                time.sleep(0.2)
            self.world = self.client.get_world()
            self.map = self.world.get_map()

        # === set fixed time-step and synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True  # Enables synchronous mode
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        self.client.reload_world(False)

        self.transform_walker_default = self.world.get_map().get_spawn_points()[35]

        # === walker ===
        self.max_walking_speed = 5   # 18/3,6 m/s
        self.walker = self.__spawn_walker()
        self.collision_sensor = None

        # === Draw Start/ End Point ===
        self._set_camara_view()

    def __spawn_walker(self):
        # === Load Blueprint and spawn walker ===
        self.walker_bp = self.blueprint_library.filter('0012')[0]
        walker = self.world.spawn_actor(
            self.walker_bp, self.transform_walker_default)
        self.actor_list.append(walker)
        try:
            self.collision_sensor = self.world.spawn_actor(
                self.blueprint_library.find('sensor.other.collision'),
                carla.Transform(), attach_to=walker)
        except:
            print("collision sensor failed")
        return walker

    def _set_camara_view(self):
        # === Walker View Camera ===
        spectator = self.world.get_spectator()
        location = self.transform_walker_default.location
        # location = carla.Location(x=143.119980, y=326.970001, z=0.300000)
        transform = carla.Transform(location, self.transform_walker_default.rotation)
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20),
                                                carla.Rotation(pitch=-90)))

    def reset_walker(self):
        try:
            self.collision_sensor.destroy()
        except:
            print("Collion Sensor")
        self.walker.destroy()
        self.walker = self.__spawn_walker()
        self.world.tick()

    def step(self, action):
        # === Let the walker do a move ===
        # action = get_round_values(action, 1)
        action_length = np.linalg.norm(action)
        if action_length == 0.0:
            # the chances are slim, but theoretically both actions could be 0.0
            unit_action = np.array([0.0, 0.0], dtype=np.float32)
        elif action_length > 1.0:
            # create a vector for the action with the length of zero
            unit_action = action / action_length
        else:
            unit_action = action

        direction = carla.Vector3D(x=float(unit_action[0]), y=float(unit_action[1]), z=0.0)
        walker_control = carla.WalkerControl(
            direction, speed=self.max_walking_speed, jump=False)
        self.walker.apply_control(walker_control)
        # === Do a tick, an check if done ===
        self.world.tick()


if __name__ == '__main__':
    env = MiniEnv()
    actions = [[0.4826754629611969, 0.875799298286438], [0.0027919672429561615, -0.7345529198646545], [-0.6582249402999878, -0.3860578238964081], [0.25546926259994507, 0.7816871404647827], [0.9266030788421631, 0.3760410249233246], [0.3238101303577423, 0.7327938079833984], [-0.9513192772865295, -0.30820727348327637], [0.008074972778558731, -0.3230511248111725], [0.9991605281829834, -0.04096579924225807], [0.7071067690849304, 0.7071067690849304], [0.5615003108978271, -0.41597673296928406], [-0.8805395364761353, 0.4739725887775421], [-0.2839803993701935, 0.9588301181793213], [0.5272359251976013, 0.8477954268455505], [-0.9470339417457581, 0.32113340497016907], [-0.2091234028339386, -0.8915301561355591], [0.4205546975135803, -0.9072672128677368], [-0.9604465961456299, 0.27846410870552063], [0.7813737392425537, -0.053366392850875854], [-0.7071067690849304, 0.7071067690849304], [0.26549065113067627, 0.6347508430480957], [0.6135621070861816, 0.6496344208717346], [0.9984074234962463, 0.05641496554017067], [0.8054863810539246, 0.5926143527030945], [0.7071067690849304, 0.7071067690849304], [-0.10058329254388809, 0.9949285984039307], [-0.036413904279470444, -0.012814752757549286], [-0.01268741488456726, 0.03003934770822525], [0.7580878138542175, -0.6521524786949158], [-0.008821874856948853, -0.3245569169521332], [-0.21454592049121857, -0.9767138957977295], [-0.9948078989982605, -0.10177096724510193], [-0.7071067690849304, 0.7071067690849304], [-0.6090502738952637, 0.782119870185852], [-0.08349954336881638, -0.15947265923023224], [-0.9879937171936035, 0.15449422597885132], [-0.15490011870861053, -0.6878093481063843], [-0.18997524678707123, -0.9817889332771301], [0.18341444432735443, -0.983035683631897], [-0.9952763319015503, 0.09708169102668762], [0.23239539563655853, -0.6906684637069702], [-0.06615660339593887, -0.13370314240455627], [0.7071067690849304, 0.7071067690849304], [0.9972230195999146, -0.07447332888841629], [0.2960509955883026, 0.2555234134197235], [0.03922887519001961, -0.9992302656173706], [-0.7071067690849304, 0.7071067690849304], [-0.7071067690849304, -0.7071067690849304], [-0.785564124584198, -0.6187803149223328], [-0.36741694808006287, 0.20105811953544617], [0.8721819519996643, 0.48918163776397705], [-0.5792128443717957, -0.8151763677597046], [-0.18572065234184265, -0.982602596282959], [-0.6306482553482056, 0.7760688662528992], [0.891381025314331, -0.4532546401023865], [-0.22308588027954102, 0.537312388420105], [-0.9852259755134583, -0.17125941812992096], [0.9671758413314819, -0.2541082501411438], [0.20528334379196167, 0.4840652048587799], [0.14745420217514038, 0.9890689253807068], [-0.32640349864959717, 0.9452306032180786], [-0.7327518463134766, 0.2820596396923065], [0.9599769711494446, 0.001070566475391388], [0.7071067690849304, -0.7071067690849304], [0.9183375835418701, 0.395797997713089], [-0.5448837876319885, 0.04242657124996185], [-0.7071067690849304, 0.7071067690849304], [-0.8638691306114197, 0.5037162899971008], [0.1464247852563858, 0.7227432727813721], [-0.010121870785951614, 0.1772642284631729], [0.25688114762306213, 0.8345487713813782], [-0.11173541843891144, 0.993738055229187], [-0.5683669447898865, -0.02837153524160385], [0.7071067690849304, 0.7071067690849304], [-0.817175030708313, 0.5763896703720093], [0.29422619938850403, 0.5256699323654175], [0.9252138137817383, 0.37944620847702026], [-0.7891393899917603, -0.08492594212293625], [-0.7486304044723511, 0.6629875302314758], [0.7125158309936523, 0.7016559839248657], [0.33038467168807983, 0.1886870265007019], [0.7496283650398254, 0.6618590354919434], [0.30667030811309814, -0.19144251942634583], [0.6378783583641052, 0.7701370716094971], [0.9860309362411499, 0.16656199097633362], [-0.7071067690849304, -0.7071067690849304], [0.596549391746521, 0.7767751216888428], [-0.589374840259552, -0.8078596591949463], [0.7071067690849304, 0.7071067690849304], [0.9938544034957886, -0.11069554835557938], [0.24722030758857727, 0.9689592719078064], [-0.7071067690849304, 0.7071067690849304], [0.2599325478076935, -0.9656267166137695], [-0.8806638121604919, 0.4737416207790375], [0.46402236819267273, 0.8858234286308289], [-0.5900712609291077, -0.32064738869667053], [0.9049132466316223, -0.42559611797332764], [-0.8467910289764404, 0.4566418528556824], [0.8908368349075317, 0.4543233811855316], [-0.6578539609909058, -0.6492488980293274], [0.7828546166419983, -0.06740579754114151], [-0.5667132139205933, 0.8239150643348694], [0.6003513336181641, -0.7489793300628662], [0.0888669341802597, 0.443553626537323], [0.7071067690849304, 0.7071067690849304], [-0.6644183397293091, -0.74736088514328], [-0.9091997742652893, 0.41636011004447937], [0.9434688687324524, -0.3314610421657562], [0.733665943145752, -0.6795102953910828], [0.9324835538864136, -0.36121246218681335], [-0.6101106405258179, 0.26860201358795166], [-0.4686035215854645, 0.8834086060523987], [-0.01863856427371502, -0.46889322996139526], [0.5092383027076721, -0.8606255650520325], [0.7906773686408997, -0.61223304271698], [-0.5617847442626953, 0.8272833824157715], [0.6478760838508606, 0.49586400389671326], [-0.7071067690849304, 0.7071067690849304], [-0.07765895128250122, 0.9969798922538757], [-0.27178725600242615, -0.317263662815094], [-0.4140666127204895, 0.9102466702461243], [-0.7721666097640991, -0.24702271819114685], [-0.40686720609664917, 0.7858258485794067], [-0.5879094004631042, 0.514868438243866], [0.28166669607162476, 0.9595123529434204], [-0.7165942192077637, 0.3446735143661499], [0.43119198083877563, -0.1404099464416504], [-0.7071067690849304, 0.7071067690849304], [-0.951020359992981, 0.3091283142566681], [0.5578708052635193, 0.31203001737594604], [0.8954235315322876, 0.4452153146266937], [0.06865957379341125, 0.9976401329040527], [-0.7071067690849304, 0.7071067690849304], [0.9342617988586426, 0.35658782720565796], [0.5005214214324951, -0.7108637690544128], [-0.16171184182167053, -0.1692361831665039], [-0.8680383563041687, -0.2642921209335327], [-0.7494245171546936, -0.6620898246765137], [-0.30248337984085083, 0.1713712513446808], [-0.47966575622558594, -0.8774513006210327], [-0.024326004087924957, 0.7163563966751099], [0.9218993186950684, 0.3874296545982361], [0.8596744537353516, -0.09112859517335892], [-0.9449474811553955, -0.32722193002700806], [-0.41880032420158386, -0.9080783724784851], [-0.3089045286178589, 0.7466543316841125], [0.03873179480433464, 0.9992496371269226], [-0.15514667332172394, 0.987891435623169], [-0.7762249708175659, -0.6304560303688049], [-0.9109365344047546, -0.11997036635875702], [0.060601651668548584, 0.9981620907783508], [-0.9138131141662598, 0.4061349928379059], [0.46726399660110474, 0.5652134418487549], [0.041250165551900864, 0.9991488456726074], [0.44586601853370667, 0.8950996398925781], [0.3481586277484894, -0.042826324701309204], [-0.9445947408676147, 0.05207543075084686], [0.18146105110645294, 0.8255013823509216], [0.5177067518234253, 0.8555580377578735], [0.6443241834640503, -0.4387170076370239], [-0.6246854066848755, 0.4152611494064331], [0.6988434791564941, 0.5626610517501831], [-0.06596527993679047, 0.8155826330184937], [0.7071067690849304, 0.7071067690849304], [0.350716769695282, -0.9364815950393677], [-0.7024087905883789, 0.7117738127708435], [0.029284894466400146, 0.4308515787124634], [0.995266318321228, 0.09718536585569382], [0.9276227951049805, 0.3735184967517853], [0.6038873195648193, 0.6435543894767761], [-0.1479184925556183, -0.9889995455741882], [0.8812682032585144, 0.4726165235042572], [0.23767384886741638, 0.030715659260749817], [0.6645083427429199, 0.2094622254371643], [-0.3118959069252014, 0.853239893913269], [0.8318452835083008, -0.0350053608417511], [-0.7248495817184448, 0.6889072060585022], [-0.7071067690849304, 0.7071067690849304], [-0.3873295485973358, -0.34216147661209106], [0.6950439214706421, 0.7189673185348511], [-0.7658093571662903, -0.1059194803237915], [-0.6730740666389465, 0.3275149464607239], [0.7071067690849304, -0.7071067690849304], [0.7071067690849304, 0.7071067690849304], [-0.8583847284317017, -0.3742825984954834], [0.7549425959587097, 0.6557909250259399], [-0.7043715715408325, -0.7098314762115479], [-0.2854897379875183, -0.9583818316459656], [0.7736195921897888, 0.6336503624916077], [0.10093393921852112, -0.9948931932449341], [0.7614108324050903, 0.5524885654449463], [0.5954267382621765, 0.8034096360206604], [-0.9309245944023132, -0.365211546421051], [-0.7071067690849304, 0.7071067690849304], [0.6347388029098511, 0.7727267742156982], [0.13333678245544434, -0.49245357513427734], [-0.4407650828361511, -0.5207672119140625], [0.7348925471305847, 0.6781835556030273], [-0.7583321928977966, -0.6518683433532715], [0.2797599732875824, 0.9402298927307129], [-0.9452083706855774, 0.3264676332473755], [-0.8765899538993835, 0.4812379777431488], [-0.028643950819969177, 0.999589741230011], [0.803986668586731, 0.42541080713272095], [0.06816752254962921, 0.9976739287376404], [0.9002663493156433, 0.4353395402431488], [0.9075489044189453, -0.4199465215206146], [-0.519519567489624, -0.21859334409236908], [-0.2313881814479828, -0.9461921453475952], [-0.0494796559214592, -0.12439128756523132], [-0.6206433773040771, 0.7840930223464966], [-0.3460223078727722, -0.7397374510765076], [-0.4936242997646332, -0.7078866362571716], [0.5789626836776733, 0.81535404920578], [-0.010263249278068542, -0.4224212169647217], [0.6195948123931885, 0.7849218249320984], [-0.9913875460624695, 0.13096053898334503], [0.9958704113960266, -0.09078602492809296], [-0.5305221676826477, 0.8476710915565491], [0.7311336398124695, 0.42688724398612976], [0.24604028463363647, -0.9692595601081848], [0.9684675335884094, -0.24913963675498962], [0.9518476724624634, 0.30657118558883667], [0.7071067690849304, -0.7071067690849304], [-0.8942421078681946, -0.4475836455821991], [-0.16979894042015076, -0.15657980740070343], [0.9998206496238708, -0.01893925666809082], [-0.7071067690849304, 0.7071067690849304], [-0.020141351968050003, -0.9997971653938293], [0.06522350758314133, -0.7147315740585327], [0.929468035697937, 0.3689028024673462], [0.9133139252662659, 0.4072563052177429], [-0.33361393213272095, -0.18457888066768646], [-0.9402773380279541, 0.34040936827659607], [-0.8980640769004822, -0.43986475467681885], [-0.41839203238487244, 0.908266544342041], [-0.9931111335754395, -0.11717619001865387], [-0.09775170683860779, -0.3059213161468506], [0.7071067690849304, -0.7071067690849304], [0.603531002998352, -0.7973395586013794], [-0.3377349376678467, -0.4453068971633911], [-0.43245750665664673, -0.9016542434692383], [0.7071067690849304, 0.7071067690849304], [-0.6471853256225586, 0.7623326778411865], [-0.5850539207458496, 0.05699129402637482], [0.881861686706543, 0.35788992047309875], [-0.7071067690849304, -0.7071067690849304], [-0.8063426613807678, -0.5191960334777832], [0.8487322330474854, -0.5288227796554565], [0.14216430485248566, -0.3326195478439331], [0.5021065473556519, 0.8648058176040649], [-0.6282835006713867, 0.2587279677391052], [0.9403241872787476, 0.3402799367904663], [-0.750303328037262, -0.6610937714576721], [0.720098614692688, -0.6066899299621582], [-0.18315228819847107, 0.7842769622802734], [0.9467213153839111, -0.19337008893489838], [0.7071067690849304, 0.7071067690849304], [-0.04095776379108429, 0.25137001276016235], [0.9936351776123047, -0.11264639347791672], [-0.8228021860122681, -0.5683279037475586], [0.6716128587722778, 0.13929279148578644], [-0.5039077401161194, -0.554779589176178], [0.16160370409488678, 0.39869481325149536], [-0.7071067690849304, -0.7071067690849304], [-0.4464830458164215, 0.8947920799255371], [-0.2852371335029602, -0.1276526302099228], [0.8497264981269836, -0.5272237062454224], [0.5962855815887451, 0.3473588228225708], [0.7071067690849304, 0.7071067690849304], [0.6294577717781067, 0.7770346403121948], [0.5983518362045288, 0.19658595323562622], [0.7071067690849304, -0.7071067690849304], [0.5733678340911865, -0.8192980289459229], [0.9564077258110046, 0.2920348048210144], [-0.5064266920089722, -0.27678418159484863], [0.5301780700683594, 0.34261631965637207], [0.7071067690849304, 0.7071067690849304], [0.6474643349647522, 0.7620958685874939], [-0.2776583731174469, -0.9606798887252808], [-0.031241759657859802, 0.04751051962375641], [0.38289675116539, 0.923791229724884], [-0.7071067690849304, 0.7071067690849304], [-0.7071067690849304, 0.7071067690849304], [0.2506779134273529, 0.5945309400558472], [0.6606545448303223, -0.7506901621818542], [0.9735270142555237, -0.22857189178466797], [0.05586753413081169, -0.14998269081115723], [0.7424274682998657, 0.6699264645576477], [-0.45552387833595276, 0.8902235627174377], [0.41434550285339355, 0.6187589764595032], [-0.3925575911998749, -0.9197273850440979], [-0.9957791566848755, -0.09178197383880615], [0.21127425134181976, 0.13191932439804077], [-0.6155970096588135, -0.09819017350673676], [0.1398022174835205, -0.5232393741607666], [0.4896283447742462, 0.6957292556762695], [0.7071067690849304, 0.7071067690849304], [0.37310591340065, -0.9277887940406799], [-0.25974076986312866, 0.613501787185669], [0.6415296196937561, 0.7670982480049133], [0.5641930103302002, 0.8256429433822632], [0.7071067690849304, 0.7071067690849304], [0.5411271452903748, 0.40842193365097046], [-0.803906261920929, -0.5947561264038086], [0.7555798292160034, -0.6550565361976624], [-0.6448748707771301, -0.7642881274223328], [-0.7203206419944763, 0.693641185760498], [-0.659837543964386, -0.1276024430990219], [-0.7071067690849304, -0.7071067690849304], [-0.9167628884315491, 0.3994317352771759], [0.5646952986717224, 0.8252994418144226], [0.5435917377471924, -0.8393498063087463], [0.5359545350074768, -0.7383867502212524], [0.9928807020187378, 0.1191125363111496], [0.6701181530952454, 0.7422544956207275], [-0.730807900428772, -0.09714378416538239], [-0.7071067690849304, 0.7071067690849304], [0.48380979895591736, -0.11471390724182129], [-0.38330668210983276, -0.5599092245101929], [0.7071067690849304, 0.7071067690849304], [-0.6169407367706299, -0.787009596824646], [-0.9797319769859314, 0.20031292736530304], [-0.14717787504196167, 0.09278716892004013], [0.12847256660461426, 0.2438545823097229], [0.5641753673553467, 0.825654923915863], [-0.01568952202796936, 0.13115011155605316], [-0.6496115922927856, 0.7602661848068237], [0.7071067690849304, 0.7071067690849304], [-0.336332768201828, 0.12036826461553574], [0.011997848749160767, -0.9999279975891113], [-0.7638126611709595, 0.6454380750656128], [0.5821149945259094, -0.8131064772605896], [0.7071067690849304, 0.7071067690849304], [-0.5937076807022095, 0.44039595127105713], [-0.7481276988983154, 0.09037117660045624], [0.9180525541305542, -0.10399271547794342], [-0.1534305214881897, 0.015082314610481262], [-0.01549915224313736, -0.44587552547454834], [-0.27041003108024597, -0.962745189666748], [0.3702707886695862, 0.2585510015487671], [0.8356776237487793, 0.45286208391189575], [0.9307003617286682, -0.36578238010406494], [-0.801121711730957, -0.598501443862915], [0.5952097773551941, 0.8035703301429749], [0.12768380343914032, -0.37969455122947693], [0.34059152007102966, 0.43166685104370117], [-0.7124537825584412, -0.14504532516002655], [0.7901484370231628, 0.21008464694023132], [0.5092393755912781, 0.8606248497962952], [0.13424751162528992, 0.3110279440879822], [-0.14451119303703308, -0.11466804146766663], [-0.4654696583747864, 0.20595750212669373], [0.12896429002285004, 0.9916492700576782], [-0.841137170791626, -0.22741886973381042], [-0.332894504070282, -0.9429640173912048], [0.7071067690849304, -0.7071067690849304], [-0.036431796848773956, 0.9755480289459229], [-0.5937729477882385, 0.8046326041221619], [-0.9339662194252014, -0.357361376285553], [-0.7699587345123291, -0.6380936503410339], [0.9830747246742249, 0.18320538103580475], [-0.2586011290550232, 0.024825185537338257], [0.21680442988872528, -0.9762150645256042], [0.09363824129104614, 0.9956062436103821], [-0.8706077337265015, 0.491977721452713], [0.9934679865837097, 0.11411093920469284], [-0.9069728851318359, -0.14984112977981567], [0.667627215385437, -0.17548349499702454], [0.7071067690849304, -0.7071067690849304], [0.5774850845336914, 0.8164013028144836], [-0.9645682573318481, 0.263833612203598], [0.3756016492843628, -0.31015974283218384], [-0.7256656885147095, -0.20301759243011475], [0.08273639529943466, 0.9965715408325195], [0.7071067690849304, -0.7071067690849304], [-0.7578451633453369, 0.6524344086647034], [-0.9252437949180603, -0.37937307357788086], [-0.6953441500663757, 0.33313804864883423], [-0.012923479080200195, 0.06795679777860641], [-0.9049606919288635, -0.4254952669143677], [0.9921337366104126, 0.12518243491649628], [0.7071067690849304, 0.7071067690849304], [-0.33079302310943604, 0.943703293800354], [0.6243256330490112, 0.1317230612039566], [-0.5249873399734497, 0.8511099815368652], [-0.31540176272392273, -0.28630051016807556], [-0.8758967518806458, -0.4824984073638916], [-0.7071067690849304, -0.7071067690849304], [0.8273633718490601, -0.5616670250892639], [-0.548645555973053, 0.670315146446228], [-0.4382788836956024, 0.898838996887207], [0.514772891998291, 0.04144638031721115], [0.7698354125022888, -0.6382423639297485], [0.5348811149597168, -0.8449273705482483], [0.6140761375427246, 0.7892467379570007], [-0.6226639747619629, -0.12314802408218384], [0.03752751648426056, -0.5032073855400085], [0.5151757597923279, 0.8570845723152161], [-0.34752410650253296, -0.6883254051208496], [-0.22486631572246552, -0.3056173622608185], [-0.7071067690849304, 0.7071067690849304], [-0.7115219235420227, -0.7026638984680176], [-0.24027691781520844, -0.39020857214927673], [-0.10727570950984955, -0.3930478096008301], [-0.851971447467804, 0.5235881805419922], [-0.7984471321105957, 0.6020649671554565], [0.7071067690849304, -0.7071067690849304], [0.5435963869094849, 0.839346706867218], [0.8854788541793823, -0.46467968821525574], [0.9308767914772034, 0.11417755484580994], [0.6819858551025391, 0.7313653230667114], [-0.17711420357227325, -0.9841902852058411], [0.6818884015083313, 0.7314561605453491], [-0.36777693033218384, 0.9299140572547913], [0.6714749932289124, -0.7410272359848022], [-0.9840695261955261, 0.17778420448303223], [-0.5566561222076416, -0.8307430148124695], [-0.17237022519111633, 0.19606220722198486], [0.42929986119270325, 0.9031620621681213], [0.2541891634464264, -0.742447018623352], [-0.24307110905647278, -0.3863193392753601], [-0.5139651894569397, 0.8578110337257385], [0.07897493988275528, -0.99687659740448], [0.6148730516433716, -0.7886260747909546], [-0.7005624175071716, 0.7135911583900452], [0.5972940921783447, 0.7465017437934875], [0.027448110282421112, 0.1586257517337799], [0.0056342557072639465, 0.3452117443084717], [-0.8341585993766785, -0.5515247583389282], [0.48320335149765015, 0.5721797943115234], [0.4292815029621124, 0.8683841228485107], [-0.08532863110303879, -0.43726006150245667], [-0.7071067690849304, -0.7071067690849304], [-0.5108920931816101, 0.43536579608917236], [-0.21410588920116425, 0.08248785138130188], [0.0008075684309005737, 0.6110836267471313], [-0.5363755226135254, -0.8439794778823853], [-0.19900356233119965, 0.838508129119873], [0.26917922496795654, -0.9630900621414185], [-0.05329284071922302, -0.6287237405776978], [0.6734695434570312, -0.7250686883926392], [0.7991888523101807, 0.5792312026023865], [-0.326432466506958, 0.9452205300331116], [0.5190190076828003, 0.7045798301696777], [0.4914548993110657, 0.5200362801551819], [-0.8220853805541992, 0.5693641901016235], [-0.2986784875392914, -0.9543538093566895], [-0.787097692489624, 0.616828441619873], [-0.14802104234695435, -0.908674955368042], [0.7071067690849304, -0.7071067690849304], [0.8835432529449463, 0.46834951639175415], [-0.9963847398757935, 0.08495555073022842], [-0.9559653997421265, 0.293479323387146], [0.30958181619644165, 0.020533353090286255], [0.1467612385749817, -0.9891720414161682], [-0.32425040006637573, 0.4221532344818115], [0.07313676178455353, -0.9973219037055969], [-0.5413520336151123, 0.8407959938049316], [-0.17900975048542023, 0.19503921270370483], [-0.5590870976448059, -0.8291088342666626], [0.8125507235527039, 0.5828904509544373], [0.37189072370529175, -0.42268121242523193], [0.789079487323761, 0.6142910122871399], [0.48339903354644775, 0.875400185585022], [-0.7522156834602356, 0.6589168906211853], [0.16558301448822021, -0.42975184321403503], [-0.7489785552024841, -0.662594199180603], [0.6480696797370911, 0.7615810632705688], [-0.7857806086540222, -0.6185052990913391], [0.744462788105011, 0.6195327043533325], [0.564452588558197, 0.6288869976997375], [0.7071067690849304, 0.7071067690849304], [-0.18627654016017914, -0.35493963956832886], [0.35754960775375366, -0.8153371810913086], [-0.7228811383247375, 0.6909723877906799], [-0.7086347341537476, 0.12388298660516739], [0.18941421806812286, 0.5556436777114868], [-0.5045261979103088, -0.7885521650314331], [-0.7989926338195801, -0.20688647031784058], [-0.99460369348526, -0.10374724864959717], [0.7071067690849304, -0.7071067690849304], [-0.7071067690849304, 0.7071067690849304], [0.7071067690849304, 0.7071067690849304], [-0.8208714723587036, -0.571112871170044], [0.4438133239746094, -0.48974451422691345], [-0.04209810867905617, 0.9991135001182556], [-0.33871111273765564, -0.940890371799469], [0.41184836626052856, 0.03325997292995453], [0.7963523268699646, 0.6048330068588257], [-0.8863213658332825, 0.4630707800388336], [0.2994216978549957, 0.49255722761154175], [0.31958258152008057, 0.2892211079597473], [0.5127456188201904, -0.8337686061859131], [0.7852461338043213, -0.6191836595535278], [-0.14344391226768494, -0.9896584153175354], [0.13353320956230164, -0.2805531919002533], [0.3856760561466217, 0.9226343631744385], [-0.7433437705039978, 0.6689096093177795], [-0.39034855365753174, -0.9206671118736267], [-0.5158810615539551, -0.8566602468490601], [0.09217189997434616, 0.04874134808778763], [-0.4671412706375122, 0.14518316090106964], [0.8747642040252686, 0.4845488965511322], [0.21606142818927765, 0.5285964608192444], [-0.43276143074035645, 0.9015085101127625], [-0.6628822684288025, 0.7487236857414246], [0.8686128854751587, 0.38911473751068115], [0.24326860904693604, -0.27293848991394043], [-0.47335466742515564, 0.8808719515800476], [-0.43565085530281067, 0.29744642972946167], [0.15539699792861938, -0.44112658500671387]]

    """ Validation behavior of a walker with fixed actions """

    # Ground truth
    env.reset_walker()
    positions = []
    for index, action in enumerate(actions):
        positions.append(env.walker.get_transform().location)
        env.step(action=action)
    for epoch in range(100):
        sprint = True
        env.reset_walker()
        for index, action in enumerate(actions):
            if positions[index].x != env.walker.get_transform().location.x:
                if sprint:
                    print("Oh no:", index, env.walker.get_transform().location.x, "soll:", positions[index].x)
                    sprint = False
            env.step(action=action)

        if epoch%10==0:
            print(epoch)
