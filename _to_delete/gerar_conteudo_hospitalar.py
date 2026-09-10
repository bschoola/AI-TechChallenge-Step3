"""Gera os protocolos internos (RAG) e as FAQs (fine-tuning) das demais
especialidades do hospital, alem da oncologia mamaria ja existente.

Conteudo sintetico do hospital ficticio "Hospital vEinstein", em nivel de
orientacao clinica geral. Nao contem posologia especifica: o assistente e de
apoio a decisao e nunca prescreve — ver agent/guardrails.py.

Uso (a partir de 1.AssistenteMedico/):
    python gerar_conteudo_hospitalar.py
"""

from pathlib import Path

BASE = Path(__file__).resolve().parent
PROTOCOLOS = BASE / "data" / "raw" / "protocolos"
FAQS = BASE / "data" / "raw" / "faqs"


# ---------------------------------------------------------------------------
# Protocolos internos — base de conhecimento do RAG (rag/ingest.py)
# ---------------------------------------------------------------------------

PROTOCOLOS_NOVOS = {
"protocolo_cli01_hipertensao": """PROTOCOLO CLI-01 — HIPERTENSAO ARTERIAL SISTEMICA

Diagnostico e classificacao: o diagnostico exige medidas em pelo menos duas consultas distintas, com tecnica adequada (paciente sentado, em repouso ha 5 minutos, manguito compativel com a circunferencia do braco). Considera-se hipertensao a partir de 140/90 mmHg em consultorio.

Avaliacao inicial: investigar lesao de orgao-alvo (funcao renal, fundo de olho, eletrocardiograma), rastrear comorbidades metabolicas e registrar todos os medicamentos em uso, incluindo anti-inflamatorios e descongestionantes, que elevam a pressao.

Paciente ja em tratamento com pressao acima da meta: antes de considerar falha terapeutica, verificar adesao ao tratamento, tecnica de medida, ingestao de sal e uso concomitante de substancias hipertensoras. Ma adesao e a causa mais frequente de descontrole aparente.

Sinais de alarme que exigem avaliacao imediata: pressao muito elevada acompanhada de dor toracica, dispneia, deficit neurologico focal, alteracao visual aguda ou rebaixamento do nivel de consciencia. Nesses casos, encaminhar a emergencia — ver Protocolo SEG-02.

Seguimento: reavaliacao em intervalo curto apos qualquer ajuste terapeutico, com reforco de medidas nao farmacologicas (restricao de sodio, atividade fisica regular, controle de peso, reducao do consumo de alcool).

Este protocolo orienta a sugestao de conduta do assistente virtual. A escolha e o ajuste de qualquer terapia sao decisao exclusiva do medico responsavel pelo caso.
""",

"protocolo_cli02_diabetes": """PROTOCOLO CLI-02 — DIABETES MELLITUS TIPO 2

Avaliacao do controle glicemico: a hemoglobina glicada reflete o controle dos ultimos tres meses e deve ser interpretada junto com o perfil de glicemias capilares. Discordancia entre glicada e glicemias sugere erro de medida, anemia ou hemoglobinopatia.

Descompensacao em paciente ja tratado: investigar, nesta ordem, adesao ao tratamento, erros de tecnica de aplicacao quando ha insulina, infeccao intercorrente, uso de corticoide e mudanca alimentar recente. Falha terapeutica real e diagnostico de exclusao.

Rastreio de complicacoes: fundo de olho anual, avaliacao de funcao renal e albuminuria anual, e exame dos pes a cada consulta, com pesquisa de sensibilidade e inspecao de lesoes. O exame dos pes e frequentemente omitido e e o de maior impacto na prevencao de amputacao.

Sinais de descompensacao aguda: poliuria e polidipsia intensas, perda de peso rapida, desidratacao, dor abdominal, respiracao profunda ou alteracao do nivel de consciencia indicam possivel emergencia hiperglicemica e exigem avaliacao imediata.

Hipoglicemia: sudorese, tremor, confusao e palpitacoes em paciente em uso de insulina ou secretagogos. Episodios repetidos exigem revisao do esquema terapeutico pelo medico assistente.

Este protocolo e informativo. Qualquer ajuste de esquema terapeutico cabe exclusivamente ao medico responsavel.
""",

"protocolo_cli03_anemia": """PROTOCOLO CLI-03 — INVESTIGACAO INICIAL DE ANEMIA

Definicao e primeiro passo: confirmada a anemia no hemograma, o volume corpuscular medio orienta a investigacao — microcitica, normocitica ou macrocitica.

Anemia microcitica: a causa mais frequente e a ferropriva. Solicitar ferritina, ferro serico e saturacao de transferrina. Ferritina baixa confirma deficiencia de ferro; ferritina normal ou alta nao exclui, pois e reagente de fase aguda.

Investigacao da causa da ferropenia: identificar a anemia nao encerra o caso — e obrigatorio buscar a origem da perda. Em mulheres em idade fertil, investigar menorragia. Em homens e em mulheres apos a menopausa, a perda pelo trato gastrointestinal deve ser investigada ativamente, incluindo avaliacao endoscopica.

Anemia macrocitica: dosar vitamina B12 e acido folico, e avaliar funcao tireoidiana, uso de alcool e medicamentos.

Sinais de alarme: anemia de instalacao rapida, instabilidade hemodinamica, sangramento ativo, dor toracica ou dispneia aos minimos esforcos exigem avaliacao de urgencia.

Seguimento: reavaliar hemograma e ferritina apos o periodo de reposicao definido pelo medico assistente, e manter a investigacao da causa mesmo apos a correcao dos indices.

Este protocolo orienta a sugestao do assistente virtual; prescricao e conduta sao do medico responsavel.
""",

"protocolo_cli04_tireoide": """PROTOCOLO CLI-04 — DISFUNCAO TIREOIDIANA

Exame inicial: o TSH e o exame de rastreio. TSH alterado deve ser complementado com T4 livre para definir se a disfuncao e clinica ou subclinica.

Hipotireoidismo primario: TSH elevado com T4 livre baixo. Quadro clinico tipico inclui fadiga, intolerancia ao frio, ganho de peso, pele seca, constipacao e bradicardia. A causa mais comum em adultos e a tireoidite autoimune — a pesquisa de anticorpos antitireoperoxidase apoia o diagnostico etiologico.

Hipotireoidismo subclinico: TSH elevado com T4 livre normal. A decisao de tratar depende do valor do TSH, da presenca de sintomas, de anticorpos positivos, da idade e do desejo de gestacao. Nao ha indicacao automatica de tratamento.

Hipertireoidismo: TSH suprimido com T4 livre elevado. Perda de peso, taquicardia, tremor, intolerancia ao calor e insonia. Exige avaliacao especializada.

Seguimento: apos inicio ou ajuste de reposicao, reavaliar TSH no intervalo definido pelo medico assistente — dosagens muito precoces levam a ajustes desnecessarios.

Situacao de atencao: gestacao ou intencao de engravidar altera as metas de TSH e exige encaminhamento prioritario.

Este protocolo e de apoio a decisao. A indicacao e o ajuste de reposicao hormonal sao do medico responsavel.
""",

"protocolo_cli05_dor_cronica": """PROTOCOLO CLI-05 — DOR CRONICA GENERALIZADA E FIBROMIALGIA

Definicao: dor musculoesqueletica difusa por mais de tres meses, tipicamente acompanhada de fadiga, sono nao reparador e queixas cognitivas.

Diagnostico: e clinico, mas de exclusao razoavel. Antes de firmar a hipotese, afastar hipotireoidismo, anemia, deficiencia de vitamina D, doencas inflamatorias e reumatologicas, e efeito adverso de medicamentos. Um painel laboratorial basico e suficiente na maioria dos casos; investigacao exaustiva repetida costuma reforcar a angustia do paciente sem alterar a conduta.

Sinais que afastam o diagnostico e exigem investigacao: febre, perda de peso, artrite objetiva com edema articular, deficit neurologico, sintomas noturnos progressivos ou alteracoes laboratoriais inflamatorias significativas.

Abordagem: o pilar do tratamento e nao farmacologico — exercicio fisico progressivo e regular, higiene do sono e abordagem psicoeducativa. O tratamento farmacologico e adjuvante e deve ser individualizado pelo medico assistente.

Comunicacao com o paciente: validar a dor como real e explicar o mecanismo de sensibilizacao central melhora a adesao. Ausencia de alteracao em exames nao significa ausencia de doenca.

Este protocolo orienta a sugestao do assistente virtual; a conduta terapeutica e do medico responsavel.
""",

"protocolo_urg01_dor_abdominal": """PROTOCOLO URG-01 — DOR ABDOMINAL AGUDA

Prioridade de triagem: dor abdominal com instabilidade hemodinamica, sinais de irritacao peritoneal, distensao com parada de eliminacao de gases e fezes, ou suspeita de gravidez ectopica e emergencia cirurgica e tem prioridade maxima.

Avaliacao dirigida: caracterizar inicio, localizacao e migracao da dor, relacao com alimentacao, sintomas associados (vomitos, febre, alteracao do habito intestinal, sintomas urinarios) e, em mulheres em idade fertil, data da ultima menstruacao.

Suspeita de apendicite aguda: dor periumbilical que migra para a fossa iliaca direita, anorexia, febre baixa e dor a descompressao. Hemograma com leucocitose apoia a hipotese, mas hemograma normal nao a exclui. A conduta e avaliacao cirurgica de urgencia, com imagem complementar quando o quadro nao for tipico. Ver Protocolo CIR-01 para encaminhamento cirurgico.

Cuidado importante: analgesia adequada nao mascara o diagnostico e nao deve ser retardada enquanto se aguarda a avaliacao cirurgica.

Exames de imagem: a escolha entre ultrassonografia e tomografia depende da hipotese, da idade e da condicao clinica. Em exame com contraste, aplicar a triagem do Protocolo SEG-04.

Reavaliacao: paciente liberado com dor abdominal sem diagnostico definido deve receber orientacao explicita de retorno imediato em caso de piora, febre ou vomitos persistentes.

Este protocolo e de apoio a decisao; a indicacao cirurgica e do medico responsavel.
""",

"protocolo_urg02_colica_renal": """PROTOCOLO URG-02 — COLICA RENAL E LITIASE URINARIA

Quadro tipico: dor lombar de inicio subito, intensa, em colica, com irradiacao para o flanco e regiao inguinal, frequentemente acompanhada de nauseas, vomitos e agitacao — o paciente nao encontra posicao antalgica. Hematuria microscopica e comum, mas sua ausencia nao exclui o diagnostico.

Avaliacao: exame de urina, funcao renal e imagem. A tomografia sem contraste e o exame de maior acuracia; a ultrassonografia e alternativa em gestantes e no seguimento.

Sinais de alarme que mudam a conduta: febre associada sugere infeccao obstrutiva e e emergencia urologica — obstrucao infectada exige desobstrucao, nao apenas antibiotico. Tambem sao criticos anuria, rim unico, dor refrataria a analgesia e vomitos incoercives.

Conduta habitual: analgesia, hidratacao e controle dos sintomas, com acompanhamento ambulatorial da eliminacao do calculo. Calculos maiores ou com obstrucao persistente exigem avaliacao urologica.

Diagnostico diferencial que nao pode ser esquecido: em paciente acima de 50 anos com dor lombar aguda, considerar aneurisma de aorta abdominal antes de assumir colica renal.

Orientacao de alta: retorno imediato em caso de febre, calafrios, reducao do volume urinario ou dor incontrolavel.

Este protocolo e informativo; prescricao e indicacao de procedimento sao do medico responsavel.
""",

"protocolo_urg03_trauma_extremidade": """PROTOCOLO URG-03 — TRAUMA DE EXTREMIDADE E SUSPEITA DE FRATURA

Avaliacao inicial: descrever o mecanismo do trauma, inspecionar deformidade, edema e equimose, e sempre avaliar e registrar o estado neurovascular distal — pulso, perfusao, sensibilidade e mobilidade dos dedos.

Emergencias ortopedicas: fratura exposta, sindrome compartimental (dor desproporcional ao achado, dor ao estiramento passivo, parestesia), luxacao e lesao vascular exigem avaliacao ortopedica imediata.

Imagem: radiografia em pelo menos duas incidencias ortogonais. Uma unica incidencia pode nao mostrar desvio. Suspeita clinica forte com radiografia normal justifica imobilizacao e reavaliacao com nova imagem, especialmente em fraturas de escafoide e do quadril.

Fratura de radio distal: mecanismo tipico e queda com a mao espalmada. Descrever desvio, cominuicao e envolvimento articular no laudo, pois esses achados definem a conduta entre tratamento conservador e cirurgico.

Conduta geral: imobilizacao provisoria, analgesia, elevacao do membro e reavaliacao ortopedica no prazo definido pela equipe. Registrar sempre a reavaliacao neurovascular apos a imobilizacao.

Orientacao de alta: retorno imediato se houver aumento da dor sob o gesso, dormencia, palidez ou frialdade dos dedos.

Este protocolo orienta a sugestao do assistente virtual. A conduta ortopedica definitiva e do medico responsavel.
""",

"protocolo_urg04_tvp": """PROTOCOLO URG-04 — SUSPEITA DE TROMBOSE VENOSA PROFUNDA

Quadro clinico: edema assimetrico de membro inferior, dor a palpacao do trajeto venoso, aumento da temperatura local e empastamento da panturrilha. Nenhum sinal isolado confirma nem exclui o diagnostico.

Estratificacao: aplicar escore de probabilidade clinica antes de solicitar exames. Em probabilidade baixa, o D-dimero negativo praticamente exclui o diagnostico. Em probabilidade moderada ou alta, o exame indicado e a ultrassonografia com doppler venoso, independentemente do D-dimero.

Fatores de risco a registrar: imobilizacao prolongada, viagem longa recente, cirurgia ou trauma recente, neoplasia ativa e tratamento oncologico, uso de anticoncepcional ou terapia hormonal, gestacao e puerperio, trombofilia e episodio previo de trombose.

Alerta critico: dispneia subita, dor toracica pleuritica, taquicardia, hipoxemia ou sincope sugerem embolia pulmonar e configuram emergencia — avaliacao imediata, sem aguardar o resultado do doppler.

Enquanto se aguarda o exame: em suspeita de alta probabilidade, a equipe medica deve avaliar a necessidade de conduta antes da confirmacao. Essa decisao e exclusivamente medica.

Paciente oncologico: o risco trombotico e elevado e o quadro pode ser oligossintomatico — manter limiar baixo de investigacao. Ver Protocolo ONCO-03.

Este protocolo e de apoio a decisao; qualquer indicacao de anticoagulacao e do medico responsavel.
""",

"protocolo_pne01_pneumonia": """PROTOCOLO PNE-01 — PNEUMONIA ADQUIRIDA NA COMUNIDADE

Suspeita clinica: tosse, febre, dispneia e dor toracica pleuritica, com ausculta revelando crepitacoes ou reducao do murmurio vesicular em um segmento pulmonar. Radiografia de torax confirma o infiltrado.

Decisao sobre local de tratamento: e a decisao mais importante do atendimento. Avaliar confusao mental, frequencia respiratoria elevada, hipotensao, idade avancada e comorbidades descompensadas. Saturacao de oxigenio em ar ambiente deve ser registrada em todos os casos.

Criterios de internacao a considerar: hipoxemia, instabilidade hemodinamica, incapacidade de aceitacao por via oral, comorbidade descompensada, ausencia de suporte domiciliar e falha do tratamento ambulatorial.

Exames complementares: em tratamento ambulatorial, radiografia e avaliacao clinica costumam bastar. Em internacao, acrescentar hemograma, funcao renal, eletrolitos e marcadores inflamatorios.

Antibioticoterapia: a escolha depende da gravidade, das comorbidades e do uso recente de antimicrobianos — ver Protocolo INF-03. A prescricao e sempre do medico responsavel.

Reavaliacao: paciente ambulatorial deve ser reavaliado em 48 a 72 horas. Ausencia de melhora nesse prazo exige revisao do diagnostico, pesquisa de complicacao (derrame pleural, abscesso) e reconsideracao do local de tratamento.

Este protocolo e informativo e nao substitui a avaliacao do medico assistente.
""",

"protocolo_pne02_asma": """PROTOCOLO PNE-02 — ASMA: CRISE E MANEJO

Avaliacao da gravidade da crise: registrar frequencia respiratoria, uso de musculatura acessoria, capacidade de falar frases completas, nivel de consciencia e saturacao de oxigenio. Pico de fluxo expiratorio, quando disponivel, ajuda a objetivar a resposta ao tratamento.

Sinais de crise grave: incapacidade de completar frases, sonolencia ou agitacao, cianose, saturacao baixa e — o achado mais perigoso — torax silencioso a ausculta, que indica fluxo aereo criticamente reduzido e nao melhora do broncoespasmo.

Conduta na crise: broncodilatador inalatorio de curta acao e corticoide sistemico precoce sao a base do tratamento, com oxigenio suplementar guiado pela saturacao. A reavaliacao apos cada ciclo de tratamento define a continuidade.

Criterios de atencao maxima: resposta incompleta apos o tratamento inicial, necessidade de doses repetidas em intervalos curtos, hipercapnia progressiva ou exaustao respiratoria exigem avaliacao imediata quanto a suporte ventilatorio.

Identificacao do desencadeante: exposicao a alergeno, infeccao respiratoria, exercicio, ma adesao a terapia de manutencao ou uso de anti-inflamatorios nao esteroidais em pacientes sensiveis.

Alta e seguimento: revisar tecnica inalatoria, verificar terapia de manutencao, fornecer plano de acao escrito e agendar reavaliacao. Crise que exige atendimento de urgencia indica controle inadequado da doenca de base.

Este protocolo e de apoio a decisao; a prescricao e do medico responsavel.
""",

"protocolo_pne03_tosse_aguda": """PROTOCOLO PNE-03 — TOSSE AGUDA E BRONQUITE

Definicao: tosse com menos de tres semanas de duracao. A grande maioria dos casos e de origem viral e autolimitada.

Bronquite aguda: tosse, inicialmente seca e depois produtiva, sem alteracao focal a ausculta e sem sinais de consolidacao. Expectoracao amarelada ou esverdeada nao indica infeccao bacteriana e, isoladamente, nao justifica antibiotico.

Quando solicitar radiografia de torax: febre persistente, taquipneia, taquicardia, hipoxemia, ausculta focal alterada, idade avancada ou comorbidade relevante. Sem esses achados, a radiografia raramente muda a conduta.

Diagnosticos a nao perder: pneumonia (ver Protocolo PNE-01), crise asmatica (ver Protocolo PNE-02), embolia pulmonar em paciente com fator de risco (ver Protocolo URG-04) e insuficiencia cardiaca descompensada.

Conduta: tratamento sintomatico, hidratacao e orientacao sobre a evolucao esperada. Explicar ao paciente que a tosse pode persistir por ate tres semanas apos a resolucao do quadro reduz retornos desnecessarios e pressao por antibiotico.

Reavaliacao: tosse que persiste alem de tres semanas, ou que se acompanha de perda de peso, sudorese noturna, hemoptise ou febre prolongada, exige investigacao adicional.

Este protocolo orienta a sugestao do assistente virtual; a prescricao e do medico responsavel.
""",

"protocolo_inf01_sindromes_febris": """PROTOCOLO INF-01 — TRIAGEM DE SINDROMES FEBRIS AGUDAS

Escopo: orientacao para diferenciar sindrome gripal, COVID-19 e dengue, que se sobrepoem clinicamente e exigem condutas distintas.

Sindrome gripal: inicio subito de febre, mialgia, cefaleia e sintomas respiratorios altos. Avaliar fatores de risco para complicacao — idade avancada, gestacao, imunossupressao e doenca cronica.

COVID-19: quadro respiratorio com contato epidemiologico conhecido. Alteracao de olfato e paladar reforca a hipotese. Registrar sempre a saturacao de oxigenio; hipoxemia pode ocorrer com pouca dispneia relatada.

Dengue: febre alta de inicio abrupto com dor retro-orbitaria, mialgia intensa, exantema e prostracao. Solicitar hemograma seriado para acompanhar plaquetas e hematocrito.

Sinais de alarme da dengue — exigem reavaliacao imediata: dor abdominal intensa e continua, vomitos persistentes, sangramento de mucosas, letargia ou irritabilidade, hipotensao postural, queda abrupta de plaquetas com elevacao do hematocrito.

Cuidado critico: em suspeita de dengue, anti-inflamatorios nao esteroidais e acido acetilsalicilico devem ser evitados pelo risco hemorragico. Essa restricao precisa ser destacada na orientacao ao paciente.

Sinais de alarme comuns a todos os quadros: dispneia, hipoxemia, confusao mental, hipotensao e desidratacao com incapacidade de ingestao oral.

Este protocolo e de apoio a decisao; a conduta e do medico responsavel.
""",

"protocolo_inf02_itu": """PROTOCOLO INF-02 — INFECCAO DO TRATO URINARIO

Cistite nao complicada: disuria, polaciuria, urgencia miccional e dor suprapubica, sem febre e sem dor lombar, em paciente sem alteracao estrutural do trato urinario. O diagnostico e clinico; o exame de urina apoia.

Pielonefrite: febre, calafrios, dor lombar e sinal de Giordano positivo. Configura infeccao alta, exige urocultura antes do inicio do tratamento e reavaliacao mais proxima.

Quando solicitar urocultura: suspeita de pielonefrite, infeccao recorrente, falha do tratamento inicial, gestacao, paciente do sexo masculino, sonda vesical e imunossupressao. Na cistite nao complicada em mulher jovem, a cultura nao e obrigatoria.

Infeccao recorrente: tres ou mais episodios em doze meses, ou dois em seis meses. Investigar fatores contribuintes — atividade sexual, uso de espermicida, estado pos-menopausa, esvaziamento vesical incompleto, litiase e diabetes descompensado. Considerar avaliacao urologica e de imagem.

Bacteriuria assintomatica: em geral nao deve ser tratada, com excecao de gestantes e de pacientes que serao submetidos a procedimento urologico invasivo. Tratar bacteriuria assintomatica sem indicacao gera resistencia sem beneficio.

Sinais de alarme: febre alta, vomitos com impossibilidade de via oral, instabilidade hemodinamica ou dor incontrolavel indicam necessidade de avaliacao hospitalar.

Este protocolo e informativo; a prescricao de antimicrobiano e do medico responsavel — ver Protocolo INF-03.
""",

"protocolo_inf03_antibioticoterapia": """PROTOCOLO INF-03 — USO RACIONAL DE ANTIMICROBIANOS

Principio geral: antimicrobiano so deve ser iniciado quando ha suspeita de infeccao bacteriana. Quadros virais — a maioria das infeccoes respiratorias altas e das gastroenterites — nao se beneficiam e expoem o paciente a efeitos adversos e a selecao de resistencia.

Antes de iniciar: registrar alergias medicamentosas de forma explicita e conferir interacoes com os medicamentos em uso — ver Protocolo SEG-01. Alergia relatada como intolerancia gastrointestinal nao e alergia verdadeira e nao deveria restringir opcoes desnecessariamente.

Coleta de culturas: sempre que possivel, coletar material para cultura antes da primeira dose, especialmente em infeccoes graves, suspeita de pielonefrite, pneumonia com criterio de internacao e falha terapeutica previa.

Reavaliacao obrigatoria em 48 a 72 horas: verificar resposta clinica e ajustar o esquema conforme o resultado da cultura, reduzindo o espectro sempre que o antibiograma permitir.

Duracao: usar o menor tempo eficaz para o quadro. Prolongar o tratamento sem indicacao aumenta o risco de efeitos adversos e de resistencia, sem ganho clinico.

Sinais de falha terapeutica: ausencia de melhora ou piora apos 72 horas exige revisao do diagnostico, pesquisa de foco nao drenado ou complicacao, e reconsideracao do agente causal.

Este protocolo e orientativo. A indicacao, a escolha e a duracao de qualquer antimicrobiano sao decisao exclusiva do medico responsavel.
""",

"protocolo_gas01_gastroenterite": """PROTOCOLO GAS-01 — GASTROENTERITE AGUDA E HIDRATACAO

Quadro tipico: diarreia aguda com ou sem vomitos, dor abdominal em colica e febre baixa, geralmente de origem viral ou alimentar e autolimitado.

Avaliacao do estado de hidratacao — o ponto central do atendimento: verificar mucosas, turgor cutaneo, frequencia cardiaca, pressao arterial com medida em pe quando possivel, diurese e nivel de consciencia.

Desidratacao leve a moderada: hidratacao por via oral com solucao de reidratacao, oferecida em pequenos volumes e frequentes. Manter a alimentacao conforme aceitacao; jejum prolongado nao e recomendado.

Indicacao de hidratacao venosa: vomitos incoercives, desidratacao grave, alteracao do nivel de consciencia ou falha da via oral.

Sinais de alarme que mudam a investigacao: sangue nas fezes, febre alta persistente, dor abdominal intensa e localizada, distensao abdominal, sinais de irritacao peritoneal ou diarreia com mais de sete dias de duracao. Ver Protocolo URG-01.

Antimicrobiano: nao esta indicado na maioria dos casos. Considerar apenas em situacoes especificas definidas pelo medico assistente — ver Protocolo INF-03.

Orientacao de alta: higiene das maos, afastamento de manipulacao de alimentos enquanto houver sintomas, e retorno imediato em caso de sinais de alarme ou incapacidade de manter hidratacao.

Este protocolo e de apoio a decisao; a conduta e do medico responsavel.
""",

"protocolo_gas02_drge": """PROTOCOLO GAS-02 — DOENCA DO REFLUXO GASTROESOFAGICO

Quadro tipico: pirose e regurgitacao, com piora apos refeicoes volumosas, ao deitar ou ao inclinar o tronco. O diagnostico e clinico na ausencia de sinais de alarme.

Sinais de alarme que exigem endoscopia: disfagia, odinofagia, perda de peso nao intencional, anemia, sangramento digestivo, vomitos persistentes, historia familiar de neoplasia do trato digestivo alto e inicio dos sintomas em idade mais avancada.

Diagnostico diferencial que nao pode ser esquecido: dor toracica de origem cardiaca pode ser confundida com pirose. Em paciente com fatores de risco cardiovascular, afastar causa coronariana antes de atribuir o sintoma ao refluxo.

Medidas nao farmacologicas: elevacao da cabeceira, evitar deitar nas duas a tres horas apos as refeicoes, reducao de peso quando indicada, e reducao de alcool e tabaco. Restricoes alimentares devem ser individualizadas conforme os gatilhos relatados pelo proprio paciente.

Tratamento farmacologico: a supressao acida e a base do tratamento, com duracao e reavaliacao definidas pelo medico assistente. Uso prolongado sem reavaliacao periodica deve ser evitado.

Falha terapeutica: ausencia de resposta apos o periodo adequado exige revisao do diagnostico e avaliacao especializada.

Este protocolo e informativo; a prescricao e do medico responsavel.
""",

"protocolo_neu01_cefaleia": """PROTOCOLO NEU-01 — CEFALEIA: SINAIS DE ALARME E ENXAQUECA

Primeira tarefa: separar cefaleia primaria de cefaleia secundaria. A anamnese dirigida aos sinais de alarme e mais util que qualquer exame de imagem indiscriminado.

Sinais de alarme que exigem investigacao imediata: cefaleia de inicio subito e intensidade maxima em segundos, cefaleia associada a febre e rigidez de nuca, deficit neurologico focal, alteracao do nivel de consciencia, papiledema, inicio apos os 50 anos, piora progressiva, piora com esforco ou manobra de Valsalva, e cefaleia em paciente imunossuprimido ou com neoplasia conhecida.

Enxaqueca: dor tipicamente unilateral, pulsatil, de intensidade moderada a forte, com nausea, fotofobia e fonofobia, agravada por atividade fisica rotineira, com ou sem aura. Diagnostico clinico; imagem nao e necessaria na ausencia de sinais de alarme.

Cefaleia por uso excessivo de medicacao: suspeitar em paciente com cefaleia em quinze ou mais dias por mes que faz uso frequente de analgesicos ou triptanos. E causa comum de cefaleia refrataria e o tratamento passa pela retirada supervisionada do agente, nao pelo aumento da dose.

Diario de cefaleia: registrar frequencia, intensidade, gatilhos e medicacoes usadas. E o instrumento mais util para orientar a decisao sobre terapia profilatica.

Este protocolo orienta a sugestao do assistente virtual; a conduta e a prescricao sao do medico responsavel.
""",

"protocolo_neu02_lombalgia": """PROTOCOLO NEU-02 — LOMBALGIA E LOMBOCIATALGIA

Classificacao inicial: separar lombalgia inespecifica, dor radicular e possivel causa especifica grave. Essa triagem define toda a conduta.

Sinais de alarme que exigem investigacao imediata: deficit neurologico progressivo, retencao ou incontinencia urinaria ou fecal, anestesia em sela (suspeita de sindrome da cauda equina — emergencia), febre, perda de peso, historia de neoplasia, trauma significativo, uso de corticoide cronico, imunossupressao e dor noturna progressiva que nao alivia com repouso.

Lombociatalgia: dor lombar irradiada pelo trajeto do nervo, abaixo do joelho, frequentemente com parestesia. A manobra de elevacao da perna estendida apoia o diagnostico. Deficit motor objetivo deve ser documentado e reavaliado.

Imagem: na lombalgia inespecifica sem sinais de alarme, exame de imagem nas primeiras semanas nao melhora o desfecho e frequentemente revela achados degenerativos assintomaticos que confundem a conduta. Indicar imagem quando ha sinal de alarme, deficit progressivo ou consideracao de intervencao.

Conduta: manter atividade dentro do tolerado — repouso prolongado no leito piora a evolucao. Analgesia, fisioterapia e orientacao postural conforme avaliacao do medico assistente.

Reavaliacao: ausencia de melhora em quatro a seis semanas, ou surgimento de qualquer sinal de alarme, exige reavaliacao e possivel encaminhamento especializado.

Este protocolo e de apoio a decisao; a conduta e do medico responsavel.
""",

"protocolo_oto01_otite_sinusite": """PROTOCOLO OTO-01 — OTITE MEDIA E SINUSITE AGUDA

Otite media aguda: otalgia de inicio recente com abaulamento e hiperemia da membrana timpanica a otoscopia. A otoscopia e indispensavel — dor de ouvido sem alteracao timpanica sugere outra causa, como disfuncao tubaria ou dor referida.

Conduta na otite: analgesia e obrigatoria em todos os casos. A indicacao de antimicrobiano depende da idade, da gravidade e da lateralidade, e e decisao do medico assistente — ver Protocolo INF-03.

Complicacoes da otite que exigem avaliacao imediata: edema, dor e hiperemia retroauricular com deslocamento do pavilhao (suspeita de mastoidite), paralisia facial, vertigem intensa ou sinais meningeos.

Sinusite aguda: obstrucao nasal, secrecao purulenta, dor ou pressao facial e reducao do olfato. A maioria dos quadros e viral.

Quando considerar origem bacteriana: sintomas por mais de dez dias sem melhora, sintomas graves com febre alta e secrecao purulenta por tres a quatro dias consecutivos, ou piora apos melhora inicial — o padrao de dupla piora.

Complicacoes da sinusite que exigem avaliacao imediata: edema periorbitario, alteracao da motricidade ocular ou da acuidade visual, cefaleia intensa, sinais neurologicos ou toxemia.

Este protocolo orienta a sugestao do assistente virtual; a prescricao e do medico responsavel.
""",

"protocolo_oft01_olho_vermelho": """PROTOCOLO OFT-01 — OLHO VERMELHO E CONJUNTIVITE

Triagem inicial: a pergunta que orienta tudo e se ha comprometimento da visao ou dor ocular verdadeira. Olho vermelho com acuidade visual preservada e sem dor profunda costuma ser conjuntival e benigno.

Sinais de alarme que exigem avaliacao oftalmologica de urgencia: reducao da acuidade visual, dor ocular intensa, fotofobia acentuada, pupila irregular ou nao reativa, opacificacao da cornea, hiperemia predominantemente perilimbica, trauma ocular, corpo estranho suspeito e uso de lente de contato com dor.

Conjuntivite bacteriana: hiperemia difusa com secrecao purulenta, palpebras aderidas ao despertar, geralmente com inicio unilateral e possivel acometimento do olho contralateral em poucos dias. Acuidade visual preservada.

Conjuntivite viral: secrecao aquosa, sensacao de corpo estranho, frequentemente associada a quadro respiratorio e a linfonodo pre-auricular palpavel. Altamente transmissivel.

Conjuntivite alergica: prurido como sintoma dominante, bilateral, com secrecao clara e historico atopico.

Medidas gerais: higiene das maos, uso de toalha individual, compressas e nao compartilhamento de objetos de uso pessoal. Nao ocluir o olho. Suspender o uso de lente de contato ate a resolucao completa.

Alerta importante: colirios com corticoide nao devem ser usados sem avaliacao oftalmologica, pelo risco de agravamento de ceratite herpetica e de elevacao da pressao intraocular.

Este protocolo e informativo; a prescricao e do medico responsavel.
""",

"protocolo_der01_dermatite": """PROTOCOLO DER-01 — DERMATITE DE CONTATO E REACOES CUTANEAS

Dermatite de contato: lesao eritematosa, pruriginosa, por vezes vesiculosa, com distribuicao que acompanha a area de exposicao ao agente — esse padrao geografico e a principal pista diagnostica.

Anamnese dirigida: identificar produtos de limpeza, cosmeticos, metais (niquel em bijuterias e fivelas), latex, plantas, medicamentos topicos e exposicoes ocupacionais. Perguntar sobre inicio em relacao a uma exposicao nova.

Conduta: afastar o agente suspeito e adotar cuidados de barreira — hidratacao da pele, uso de luvas na exposicao inevitavel e evitar sabonetes irritantes. O tratamento topico e definido pelo medico assistente.

Sinais de alarme que exigem avaliacao imediata: acometimento extenso, envolvimento de mucosas, bolhas com descolamento cutaneo, febre e toxemia — quadros compativeis com reacao cutanea grave a medicamento.

Reacao alergica sistemica: urticaria disseminada com angioedema, dispneia, estridor, hipotensao ou sintomas gastrointestinais configura anafilaxia e e emergencia com necessidade de atendimento imediato.

Registro obrigatorio: toda reacao cutanea atribuida a medicamento deve ser registrada de forma explicita no campo de alergias do prontuario, com o nome do agente e a descricao da reacao — ver Protocolo SEG-01.

Este protocolo orienta a sugestao do assistente virtual; a conduta e do medico responsavel.
""",

"protocolo_cir01_hernia_inguinal": """PROTOCOLO CIR-01 — HERNIA INGUINAL E AVALIACAO CIRURGICA ELETIVA

Diagnostico: abaulamento na regiao inguinal que aumenta com esforco, tosse ou ortostase e reduz em decubito. O exame deve ser feito em pe e deitado, com manobra de esforco.

Classificacao operacional: redutivel (retorna a cavidade), encarcerada (nao redutivel, sem sofrimento vascular) ou estrangulada (comprometimento vascular do conteudo).

Emergencia cirurgica: hernia irredutivel acompanhada de dor intensa, hiperemia local, vomitos, distensao abdominal e parada de eliminacao de gases e fezes sugere estrangulamento ou obstrucao intestinal — encaminhamento imediato, sem tentativa repetida de reducao forcada. Ver Protocolo URG-01.

Hernia nao complicada: conduta eletiva. O encaminhamento cirurgico considera sintomas, tamanho, ocupacao do paciente, risco de encarceramento e comorbidades.

Avaliacao pre-operatoria: registrar comorbidades, medicamentos em uso — em especial anticoagulantes e antiagregantes — alergias e exames pendentes relevantes. Ver Protocolo SEG-03.

Orientacao ao paciente enquanto aguarda a cirurgia: procurar atendimento imediato se o abaulamento deixar de reduzir, se houver dor intensa, vomitos ou parada de eliminacao de gases e fezes.

Este protocolo e de apoio a decisao; a indicacao cirurgica e do cirurgiao responsavel.
""",

"protocolo_seg01_alergias_interacoes": """PROTOCOLO SEG-01 — ALERGIAS E INTERACOES MEDICAMENTOSAS

Registro de alergias: toda alergia deve ser documentada com o nome do agente e a descricao objetiva da reacao. "Alergia a antibiotico" sem especificacao e um registro incompleto que restringe opcoes terapeuticas desnecessariamente.

Diferenciacao essencial: intolerancia (nausea, desconforto gastrico) nao e alergia. Reacao alergica verdadeira envolve urticaria, angioedema, broncoespasmo ou anafilaxia. Confundir as duas leva ao uso de alternativas de espectro mais amplo e maior toxicidade sem necessidade.

Conferencia antes de qualquer sugestao terapeutica: o assistente deve verificar o campo de alergias e a lista de medicamentos em uso do paciente antes de mencionar qualquer classe terapeutica, e sinalizar explicitamente conflitos encontrados.

Interacoes de atencao prioritaria: anticoagulantes com anti-inflamatorios e com antimicrobianos; antiagregantes em pacientes com procedimento programado; medicamentos que prolongam o intervalo QT em associacao; e uso concomitante de multiplos depressores do sistema nervoso central.

Polifarmacia: pacientes com cinco ou mais medicamentos de uso continuo tem risco elevado de interacao e de cascata terapeutica. Recomenda-se revisao periodica da lista completa, incluindo fitoterapicos e suplementos, que costumam ser omitidos pelo paciente.

Registro de reacao adversa: qualquer reacao adversa observada deve ser registrada no prontuario e comunicada a equipe.

Este protocolo e informativo. A decisao sobre qualquer terapia e do medico responsavel.
""",

"protocolo_seg02_sinais_vitais": """PROTOCOLO SEG-02 — SINAIS VITAIS DE ALERTA E ESCALONAMENTO

Objetivo: padronizar os valores que exigem sinalizacao imediata a equipe medica, em qualquer setor e para qualquer especialidade.

Valores de alerta que devem ser sinalizados:
- Frequencia respiratoria elevada ou muito reduzida
- Saturacao de oxigenio em ar ambiente abaixo de 94 por cento, ou queda em relacao ao basal do paciente
- Pressao arterial sistolica baixa, ou queda significativa em relacao ao basal
- Frequencia cardiaca persistentemente elevada ou bradicardia sintomatica
- Temperatura elevada com calafrios, ou hipotermia
- Alteracao aguda do nivel de consciencia ou do comportamento habitual

Combinacoes de maior gravidade: taquipneia com hipotensao, ou febre com alteracao do nivel de consciencia, sugerem deterioracao clinica e sepse — exigem avaliacao medica imediata, nao apenas registro.

Contexto importa: um valor isoladamente alterado em paciente estavel tem peso diferente de uma tendencia de piora ao longo das medidas. A comparacao com os valores anteriores do proprio paciente e mais informativa que o valor absoluto.

Registro: toda alteracao sinalizada deve constar no prontuario com horario, valor e conduta adotada.

Papel do assistente virtual: sinalizar valores fora da faixa esperada e destacar tendencias, sempre como alerta a ser avaliado pela equipe — nunca como decisao clinica autonoma.

Este protocolo e de apoio; a avaliacao e a conduta sao do medico responsavel.
""",

"protocolo_seg03_exames_pendentes": """PROTOCOLO SEG-03 — EXAMES PENDENTES E CONTINUIDADE DO CUIDADO

Principio: exame solicitado e nao avaliado e uma das principais fontes de falha assistencial evitavel. Todo exame pendente deve ser rastreado ate ter resultado avaliado e registrado.

Sinalizacao obrigatoria: ao abrir o prontuario de um paciente com exame pendente, a equipe deve ser alertada sobre o tipo do exame, a data da solicitacao e o medico solicitante, antes de qualquer decisao de conduta.

Exames de prioridade critica: biopsia e anatomopatologico, imagem solicitada para investigacao de suspeita de neoplasia, culturas em paciente em uso de antimicrobiano, exames de imagem em suspeita de trombose ou embolia, e reavaliacao de imagem em paciente oncologico em tratamento.

Antes de sugerir qualquer conduta: verificar se ha exame pendente que possa alterar a decisao. Sugerir conduta definitiva ignorando um exame diagnostico em andamento e um erro de raciocinio clinico.

Resultado alterado: resultado com achado critico exige comunicacao ativa ao medico solicitante, e nao apenas o registro no sistema.

Transicao de cuidado: na alta ou na transferencia, os exames pendentes devem ser listados explicitamente no documento de alta, com a indicacao de quem fara a avaliacao do resultado.

Este protocolo e informativo; a avaliacao do resultado e a conduta sao do medico responsavel.
""",

"protocolo_adm01_checkup": """PROTOCOLO ADM-01 — CONSULTA DE ROTINA E RASTREAMENTO

Escopo: orientacao para a consulta de check-up de paciente assintomatico, em que o objetivo e o rastreamento e a prevencao, nao a investigacao de queixa.

Principio: rastrear e diferente de investigar. Exames indiscriminados em paciente assintomatico produzem achados incidentais que geram ansiedade, investigacao adicional e risco, sem beneficio demonstrado.

Elementos da avaliacao de rotina: historia familiar, habitos (tabagismo, alcool, atividade fisica, alimentacao, sono), medicamentos e suplementos em uso, saude mental, situacao vacinal, medida de pressao arterial, peso e altura.

Rastreamentos consolidados a considerar conforme idade, sexo e fatores de risco: risco cardiovascular e perfil metabolico, rastreamento de cancer colorretal, rastreamento de cancer de colo uterino, rastreamento mamografico (ver Protocolo MAMA-01), e avaliacao de risco osseo quando indicado.

Aconselhamento: as intervencoes de maior impacto na consulta de rotina sao a cessacao do tabagismo, a atividade fisica regular, a atualizacao vacinal e o controle de pressao e glicemia — mais relevantes que a ampliacao do painel laboratorial.

Registro: mesmo sem achados, a consulta de rotina deve registrar o que foi rastreado e a data, para orientar o intervalo do proximo rastreamento.

Este protocolo orienta a sugestao do assistente virtual; a indicacao de exames e do medico responsavel.
""",
}


# ---------------------------------------------------------------------------
# FAQs — pares instrucao->resposta para o fine-tuning
# Prefixo hvn_ = Hospital vEinstein (conteudo sintetico interno), para
# distinguir das FAQs derivadas de MedQuAD/PubMedQA (prefixos cancergov_,
# pubmedqa_, seniorhealth_, gard_, ghr_, mplustopics_).
# ---------------------------------------------------------------------------

FAQS_NOVAS = {
# --- Clinica medica -------------------------------------------------------
"hvn_hipertensao_descompensada": (
"Paciente hipertenso em tratamento chega com pressao bem acima da meta. Como conduzir a investigacao?",
"Antes de assumir falha terapeutica, o Protocolo CLI-01 recomenda verificar as causas mais frequentes de descontrole aparente: adesao ao tratamento, tecnica de medida da pressao, ingestao de sal e uso de substancias hipertensoras, como anti-inflamatorios e descongestionantes. Ma adesao e a causa mais comum. Vale tambem reavaliar lesao de orgao-alvo com funcao renal, fundo de olho e eletrocardiograma. Se a pressao muito elevada vier acompanhada de dor toracica, dispneia, deficit neurologico focal ou alteracao visual aguda, o quadro deixa de ser ambulatorial e exige avaliacao de emergencia. Qualquer ajuste de esquema terapeutico e decisao do medico responsavel pelo caso."),

"hvn_hipertensao_medida_correta": (
"Qual a tecnica correta de medida da pressao arterial para nao superestimar o valor?",
"O Protocolo CLI-01 orienta que a medida seja feita com o paciente sentado, apos pelo menos cinco minutos de repouso, com as costas apoiadas, os pes no chao e o braco na altura do coracao, usando manguito compativel com a circunferencia do braco. Manguito estreito demais para o braco superestima o valor de forma significativa. O diagnostico de hipertensao nao se faz com uma unica afericao: exige medidas em pelo menos duas consultas distintas. Quando ha duvida entre hipertensao verdadeira e efeito do ambiente de consultorio, a monitorizacao residencial ou ambulatorial pode ser considerada pelo medico assistente."),

"hvn_diabetes_descompensado": (
"Paciente com diabetes tipo 2 apresenta hemoglobina glicada bem acima da meta. Como abordar?",
"Conforme o Protocolo CLI-02, falha terapeutica real e diagnostico de exclusao. A investigacao deve percorrer, nesta ordem, a adesao ao tratamento, erros de tecnica de aplicacao quando ha insulina, infeccao intercorrente, uso de corticoide e mudanca alimentar recente. Vale confrontar a glicada com o perfil de glicemias capilares: discordancia entre as duas sugere erro de medida, anemia ou hemoglobinopatia. Na mesma consulta, e recomendavel rastrear complicacoes — fundo de olho, funcao renal com albuminuria e exame dos pes. Sintomas de descompensacao aguda, como perda de peso rapida, dor abdominal ou alteracao do nivel de consciencia, exigem avaliacao imediata. O ajuste terapeutico cabe ao medico responsavel."),

"hvn_diabetes_exame_pes": (
"Por que o exame dos pes e destacado no seguimento do paciente diabetico?",
"O Protocolo CLI-02 recomenda o exame dos pes a cada consulta, com pesquisa de sensibilidade e inspecao de lesoes, porque e o item de maior impacto na prevencao de amputacao e, ao mesmo tempo, o mais frequentemente omitido na pratica. A neuropatia periferica faz com que lesoes evoluam sem dor, e o paciente pode nao relatar espontaneamente. A avaliacao inclui inspecao de pontos de pressao, micoses interdigitais, unhas, deformidades e pulsos distais. Achados alterados devem ser registrados e discutidos com o medico assistente para definicao de conduta e eventual encaminhamento."),

"hvn_anemia_investigacao": (
"Paciente com anemia microcitica e ferritina baixa. Basta iniciar reposicao de ferro?",
"Nao. O Protocolo CLI-03 e explicito: identificar a anemia ferropriva nao encerra o caso — e obrigatorio investigar a origem da perda. Em mulheres em idade fertil, a menorragia e causa frequente e deve ser questionada de forma dirigida. Em homens e em mulheres apos a menopausa, a perda pelo trato gastrointestinal precisa ser investigada ativamente, incluindo avaliacao endoscopica. Vale lembrar que ferritina normal ou elevada nao exclui deficiencia de ferro, por ser reagente de fase aguda. O seguimento inclui reavaliacao de hemograma e ferritina apos o periodo de reposicao. A indicacao e a duracao da reposicao sao definidas pelo medico responsavel."),

"hvn_hipotireoidismo_subclinico": (
"TSH elevado com T4 livre normal em paciente com poucos sintomas. Ha indicacao automatica de tratar?",
"Nao ha indicacao automatica. O Protocolo CLI-04 classifica esse achado como hipotireoidismo subclinico, e a decisao de tratar depende do valor do TSH, da presenca de sintomas, da positividade de anticorpos antitireoperoxidase, da idade do paciente e do desejo de gestacao. Uma situacao muda a conduta de forma importante: gestacao ou intencao de engravidar altera as metas de TSH e justifica encaminhamento prioritario. Vale confirmar o resultado antes de qualquer decisao, ja que elevacoes transitorias de TSH ocorrem apos doenca aguda. A indicacao de reposicao hormonal e do medico responsavel."),

"hvn_fibromialgia_diagnostico": (
"Paciente com dor difusa ha meses e exames normais. Como conduzir a hipotese de fibromialgia?",
"O Protocolo CLI-05 trata o diagnostico como clinico, mas de exclusao razoavel: antes de firma-lo, convem afastar hipotireoidismo, anemia, deficiencia de vitamina D, doencas inflamatorias e efeito adverso de medicamentos, com um painel laboratorial basico. Investigacao exaustiva repetida costuma reforcar a angustia do paciente sem mudar a conduta. Achados que afastam a hipotese e pedem investigacao sao febre, perda de peso, artrite objetiva com edema, deficit neurologico e alteracoes inflamatorias laboratoriais. O pilar do tratamento e nao farmacologico — exercicio progressivo, higiene do sono e abordagem psicoeducativa. Validar a dor como real melhora a adesao: exame normal nao significa ausencia de doenca. A conduta terapeutica e do medico responsavel."),

# --- Urgencia -------------------------------------------------------------
"hvn_apendicite_suspeita": (
"Paciente com dor que migrou para fossa iliaca direita e leucocitose. Qual a conduta?",
"O quadro descrito e sugestivo de apendicite aguda, e o Protocolo URG-01 indica avaliacao cirurgica de urgencia. Os elementos que reforcam a hipotese sao a migracao da dor da regiao periumbilical para a fossa iliaca direita, anorexia, febre baixa e dor a descompressao. Um ponto importante: hemograma normal nao exclui o diagnostico, entao o raciocinio nao deve ser abandonado por um exame laboratorial tranquilizador. Imagem complementar e util quando o quadro nao e tipico. Vale destacar que analgesia adequada nao mascara o diagnostico e nao deve ser retardada enquanto se aguarda a avaliacao cirurgica. A indicacao operatoria e do cirurgiao responsavel."),

"hvn_dor_abdominal_alarme": (
"Quais achados em dor abdominal aguda indicam prioridade maxima na triagem?",
"Segundo o Protocolo URG-01, tem prioridade maxima a dor abdominal acompanhada de instabilidade hemodinamica, sinais de irritacao peritoneal, distensao com parada de eliminacao de gases e fezes, ou suspeita de gravidez ectopica em mulher em idade fertil. Por isso a data da ultima menstruacao deve ser sempre questionada nesse cenario. A avaliacao dirigida caracteriza inicio, localizacao e migracao da dor, relacao com alimentacao e sintomas associados. Quando o paciente e liberado sem diagnostico definido, a orientacao explicita de retorno imediato em caso de piora, febre ou vomitos persistentes faz parte da conduta. A decisao clinica cabe ao medico responsavel."),

"hvn_colica_renal": (
"Paciente com dor lombar em colica e hematuria microscopica. O que nao pode passar despercebido?",
"O Protocolo URG-02 destaca dois pontos. O primeiro: febre associada a colica renal sugere obstrucao infectada, que e emergencia urologica — nesse cenario, o tratamento nao e apenas antimicrobiano, e sim desobstrucao. Tambem sao criticos anuria, rim unico, dor refrataria a analgesia e vomitos incoercives. O segundo ponto e o diagnostico diferencial: em paciente acima de 50 anos com dor lombar aguda, e preciso considerar aneurisma de aorta abdominal antes de assumir colica renal. Vale lembrar que a ausencia de hematuria nao exclui litiase. A tomografia sem contraste e o exame de maior acuracia. A conduta e do medico responsavel."),

"hvn_fratura_avaliacao": (
"Paciente com queda sobre a mao espalmada e dor em punho. O que registrar na avaliacao?",
"O Protocolo URG-03 orienta descrever o mecanismo do trauma e, obrigatoriamente, avaliar e registrar o estado neurovascular distal — pulso, perfusao, sensibilidade e mobilidade dos dedos — antes e depois de qualquer imobilizacao. A imagem deve incluir pelo menos duas incidencias ortogonais, ja que uma unica incidencia pode nao mostrar desvio. No caso de fratura de radio distal, o laudo precisa descrever desvio, cominuicao e envolvimento articular, porque sao esses achados que definem a escolha entre tratamento conservador e cirurgico. Suspeita clinica forte com radiografia normal justifica imobilizacao e nova imagem na reavaliacao. A conduta ortopedica definitiva e do medico responsavel."),

"hvn_sindrome_compartimental": (
"Quais sinais devem levantar suspeita de sindrome compartimental apos trauma de extremidade?",
"O Protocolo URG-03 classifica a sindrome compartimental como emergencia ortopedica. O sinal mais precoce e mais confiavel e a dor desproporcional ao achado do exame, tipicamente agravada pelo estiramento passivo dos musculos do compartimento. Parestesia distal tambem e precoce. Sinais como ausencia de pulso e palidez sao tardios e a sua presenca ja indica quadro avancado — esperar por eles para levantar a suspeita e um erro. Em paciente imobilizado, aumento progressivo da dor sob o gesso exige reavaliacao imediata. A avaliacao e a conduta sao do ortopedista responsavel."),

"hvn_tvp_suspeita": (
"Paciente com edema assimetrico e dor em panturrilha. Como investigar trombose venosa profunda?",
"O Protocolo URG-04 orienta aplicar um escore de probabilidade clinica antes de solicitar exames. Em probabilidade baixa, D-dimero negativo praticamente exclui o diagnostico. Em probabilidade moderada ou alta, o exame indicado e a ultrassonografia com doppler venoso, independentemente do D-dimero. Devem ser registrados os fatores de risco: imobilizacao, viagem longa, cirurgia ou trauma recente, neoplasia ativa e tratamento oncologico, uso de hormonio, gestacao, trombofilia e episodio previo. O alerta critico e a embolia pulmonar: dispneia subita, dor toracica pleuritica, taquicardia, hipoxemia ou sincope exigem avaliacao imediata, sem aguardar o doppler. Qualquer indicacao de anticoagulacao e do medico responsavel."),

# --- Pneumologia ----------------------------------------------------------
"hvn_pneumonia_internacao": (
"Paciente com pneumonia adquirida na comunidade. Como decidir entre tratamento ambulatorial e internacao?",
"O Protocolo PNE-01 considera essa a decisao mais importante do atendimento. Os elementos a avaliar sao confusao mental, frequencia respiratoria elevada, hipotensao, idade avancada e comorbidades descompensadas, e a saturacao de oxigenio em ar ambiente deve ser registrada em todos os casos. Pesam a favor da internacao hipoxemia, instabilidade hemodinamica, incapacidade de aceitacao por via oral, comorbidade descompensada e ausencia de suporte domiciliar. Em tratamento ambulatorial, a reavaliacao em 48 a 72 horas nao e opcional: ausencia de melhora nesse prazo exige revisao do diagnostico, pesquisa de complicacao como derrame pleural ou abscesso, e reconsideracao do local de tratamento. A prescricao e do medico responsavel."),

"hvn_crise_asmatica": (
"Quais sinais indicam gravidade em uma crise asmatica?",
"O Protocolo PNE-02 orienta registrar frequencia respiratoria, uso de musculatura acessoria, capacidade de falar frases completas, nivel de consciencia e saturacao de oxigenio. Indicam crise grave a incapacidade de completar frases, sonolencia ou agitacao, cianose e saturacao baixa. O achado mais perigoso e o torax silencioso a ausculta: a ausencia de sibilos nesse contexto significa fluxo aereo criticamente reduzido, e nao melhora do broncoespasmo — interpretar isso como evolucao favoravel e um erro grave. Resposta incompleta apos o tratamento inicial, necessidade de doses repetidas em intervalos curtos ou hipercapnia progressiva exigem avaliacao imediata quanto a suporte ventilatorio. A conduta e do medico responsavel."),

"hvn_asma_alta": (
"O que deve ser revisado antes da alta de um paciente que teve crise asmatica?",
"O Protocolo PNE-02 orienta revisar a tecnica inalatoria, verificar a terapia de manutencao em uso, fornecer plano de acao escrito e agendar reavaliacao. Vale identificar e registrar o desencadeante da crise — exposicao a alergeno, infeccao respiratoria, exercicio, ma adesao a manutencao ou uso de anti-inflamatorios nao esteroidais em pacientes sensiveis. Um ponto conceitual importante: crise que exigiu atendimento de urgencia e, por si so, indicador de controle inadequado da doenca de base, e nao apenas um evento isolado. Isso deve orientar a revisao do tratamento de manutencao pelo medico assistente."),

"hvn_bronquite_antibiotico": (
"Paciente com tosse produtiva e secrecao esverdeada ha cinco dias. Isso indica antibiotico?",
"Nao isoladamente. O Protocolo PNE-03 e explicito: a coloracao amarelada ou esverdeada da expectoracao nao indica infeccao bacteriana e, sozinha, nao justifica antimicrobiano. A bronquite aguda e majoritariamente viral e autolimitada, com ausculta sem alteracao focal e sem sinais de consolidacao. A radiografia de torax passa a ser util quando ha febre persistente, taquipneia, taquicardia, hipoxemia, ausculta focal alterada, idade avancada ou comorbidade relevante. Explicar ao paciente que a tosse pode persistir por ate tres semanas apos a resolucao do quadro reduz retornos e pressao por antibiotico. Ver tambem o Protocolo INF-03. A prescricao e do medico responsavel."),

# --- Infectologia ---------------------------------------------------------
"hvn_dengue_sinais_alarme": (
"Quais sao os sinais de alarme da dengue que exigem reavaliacao imediata?",
"O Protocolo INF-01 lista dor abdominal intensa e continua, vomitos persistentes, sangramento de mucosas, letargia ou irritabilidade, hipotensao postural e queda abrupta de plaquetas com elevacao do hematocrito. Esses sinais tipicamente surgem no periodo de defervescencia, quando o paciente parece estar melhorando da febre — por isso a orientacao de retorno precisa ser explicita. Ha ainda um cuidado critico de conduta: em suspeita de dengue, anti-inflamatorios nao esteroidais e acido acetilsalicilico devem ser evitados pelo risco hemorragico, e essa restricao precisa ser destacada na orientacao ao paciente. O hemograma seriado acompanha plaquetas e hematocrito. A conduta e do medico responsavel."),

"hvn_sindrome_gripal_covid": (
"Como diferenciar sindrome gripal, COVID-19 e dengue na triagem inicial?",
"O Protocolo INF-01 reconhece que os tres quadros se sobrepoem clinicamente. A sindrome gripal cursa com inicio subito de febre, mialgia, cefaleia e sintomas respiratorios altos. A COVID-19 e sugerida por quadro respiratorio com contato epidemiologico conhecido, e a alteracao de olfato e paladar reforca a hipotese; a saturacao de oxigenio deve ser sempre registrada, porque a hipoxemia pode ocorrer com pouca dispneia relatada. A dengue se caracteriza por febre alta abrupta com dor retro-orbitaria, mialgia intensa, exantema e prostracao, sem sintomas respiratorios proeminentes, e pede hemograma seriado. Sao comuns a todos os quadros os sinais de alarme de dispneia, hipoxemia, confusao mental, hipotensao e desidratacao. A conduta e do medico responsavel."),

"hvn_itu_urocultura": (
"Em quais situacoes a urocultura e necessaria na infeccao urinaria?",
"O Protocolo INF-02 indica urocultura em suspeita de pielonefrite, infeccao recorrente, falha do tratamento inicial, gestacao, paciente do sexo masculino, presenca de sonda vesical e imunossupressao. Na cistite nao complicada em mulher jovem, com disuria, polaciuria e dor suprapubica sem febre nem dor lombar, o diagnostico e clinico e a cultura nao e obrigatoria. Quando a pielonefrite e a hipotese, a cultura deve ser colhida antes do inicio do tratamento — ver Protocolo INF-03. Febre alta, vomitos com impossibilidade de via oral ou instabilidade hemodinamica indicam necessidade de avaliacao hospitalar. A prescricao e do medico responsavel."),

"hvn_itu_recorrente": (
"O que investigar em paciente com infeccao urinaria de repeticao?",
"O Protocolo INF-02 define infeccao recorrente como tres ou mais episodios em doze meses, ou dois em seis meses. A investigacao busca fatores contribuintes: atividade sexual, uso de espermicida, estado pos-menopausa, esvaziamento vesical incompleto, litiase urinaria e diabetes descompensado. Avaliacao urologica e exame de imagem podem ser considerados. Um ponto de seguranca frequentemente confundido: bacteriuria assintomatica em geral nao deve ser tratada, com excecao de gestantes e de pacientes que serao submetidos a procedimento urologico invasivo — tratar sem indicacao gera resistencia sem beneficio. A conduta e do medico responsavel."),

"hvn_antibiotico_uso_racional": (
"Quais cuidados o hospital orienta antes de iniciar um antimicrobiano?",
"O Protocolo INF-03 orienta, antes da primeira dose, registrar alergias de forma explicita e conferir interacoes com os medicamentos em uso, conforme o Protocolo SEG-01. Sempre que possivel, coletar material para cultura antes de iniciar, especialmente em infeccoes graves, suspeita de pielonefrite, pneumonia com criterio de internacao e falha terapeutica previa. A reavaliacao em 48 a 72 horas e obrigatoria, para verificar resposta clinica e reduzir o espectro conforme o antibiograma. A duracao deve ser a menor eficaz para o quadro. Ausencia de melhora apos 72 horas exige revisao do diagnostico, pesquisa de foco nao drenado e reconsideracao do agente. A indicacao, a escolha e a duracao sao decisao exclusiva do medico responsavel."),

"hvn_alergia_antibiotico_registro": (
"Paciente relata alergia a antibiotico sem especificar qual. Como proceder?",
"O Protocolo SEG-01 considera esse um registro incompleto, que restringe opcoes terapeuticas desnecessariamente. A orientacao e detalhar o agente e a descricao objetiva da reacao. A diferenciacao mais importante e entre intolerancia e alergia verdadeira: nausea e desconforto gastrico caracterizam intolerancia, enquanto urticaria, angioedema, broncoespasmo e anafilaxia caracterizam reacao alergica. Confundir as duas leva ao uso de alternativas de espectro mais amplo e maior toxicidade sem necessidade. O prontuario deve ser atualizado com o nome do agente e a reacao descrita. A decisao sobre a terapia a ser adotada e do medico responsavel."),

# --- Gastroenterologia ----------------------------------------------------
"hvn_gastroenterite_hidratacao": (
"Paciente com gastroenterite aguda e desidratacao leve. Qual a abordagem de hidratacao?",
"O Protocolo GAS-01 coloca a avaliacao do estado de hidratacao como ponto central do atendimento: mucosas, turgor cutaneo, frequencia cardiaca, pressao arterial com medida em pe quando possivel, diurese e nivel de consciencia. Na desidratacao leve a moderada, a via oral com solucao de reidratacao, em pequenos volumes e frequentes, e a conduta indicada, mantendo alimentacao conforme aceitacao — jejum prolongado nao e recomendado. A hidratacao venosa fica reservada a vomitos incoercives, desidratacao grave, alteracao do nivel de consciencia ou falha da via oral. Antimicrobiano nao esta indicado na maioria dos casos. A orientacao de alta inclui retorno imediato se houver sinais de alarme. A conduta e do medico responsavel."),

"hvn_diarreia_alarme": (
"Quais achados em diarreia aguda indicam que nao se trata de quadro autolimitado?",
"O Protocolo GAS-01 destaca sangue nas fezes, febre alta persistente, dor abdominal intensa e localizada, distensao abdominal, sinais de irritacao peritoneal e diarreia com mais de sete dias de duracao. Esses achados afastam a suposicao de gastroenterite viral autolimitada e pedem investigacao adicional, com atencao ao diagnostico diferencial de abdome agudo — ver Protocolo URG-01. Vale registrar tambem uso recente de antimicrobiano, viagem recente e outros casos no mesmo domicilio ou local de trabalho, que orientam a hipotese etiologica. A investigacao e a conduta sao definidas pelo medico responsavel."),

"hvn_drge_sinais_alarme": (
"Paciente com pirose e regurgitacao ha meses. Quando indicar endoscopia?",
"O Protocolo GAS-02 trata o diagnostico como clinico na ausencia de sinais de alarme. Indicam endoscopia disfagia, odinofagia, perda de peso nao intencional, anemia, sangramento digestivo, vomitos persistentes, historia familiar de neoplasia do trato digestivo alto e inicio dos sintomas em idade mais avancada. Ha ainda um diagnostico diferencial que nao pode ser esquecido: dor toracica de origem cardiaca pode ser confundida com pirose, e em paciente com fatores de risco cardiovascular a causa coronariana deve ser afastada antes de atribuir o sintoma ao refluxo. As medidas nao farmacologicas incluem elevacao da cabeceira e evitar deitar nas duas a tres horas apos as refeicoes. A prescricao e do medico responsavel."),

# --- Neurologia -----------------------------------------------------------
"hvn_cefaleia_alarme": (
"Quais sinais de alarme em cefaleia exigem investigacao imediata?",
"O Protocolo NEU-01 lista cefaleia de inicio subito com intensidade maxima em segundos, cefaleia com febre e rigidez de nuca, deficit neurologico focal, alteracao do nivel de consciencia, papiledema, inicio apos os 50 anos, piora progressiva, piora com esforco ou manobra de Valsalva, e cefaleia em paciente imunossuprimido ou com neoplasia conhecida. A anamnese dirigida a esses sinais e mais util do que solicitar imagem de forma indiscriminada: na ausencia deles, a cefaleia primaria e o cenario mais provavel e a imagem raramente altera a conduta. A investigacao e a conduta sao definidas pelo medico responsavel."),

"hvn_cefaleia_uso_excessivo": (
"Paciente com enxaqueca refrataria que usa analgesico quase diariamente. O que considerar?",
"O Protocolo NEU-01 orienta suspeitar de cefaleia por uso excessivo de medicacao em paciente com dor em quinze ou mais dias por mes e uso frequente de analgesicos ou triptanos. E uma causa comum de cefaleia aparentemente refrataria, e o ponto contraintuitivo e que o tratamento passa pela retirada supervisionada do agente, nao pelo aumento da dose ou pela troca por outro sintomatico. O diario de cefaleia, registrando frequencia, intensidade, gatilhos e medicacoes usadas, e o instrumento mais util para orientar a decisao sobre terapia profilatica. A conduta de retirada e a indicacao de profilaxia sao definidas pelo medico responsavel."),

"hvn_lombalgia_imagem": (
"Paciente com lombalgia ha duas semanas, sem deficit neurologico. Deve fazer ressonancia?",
"O Protocolo NEU-02 orienta que, na lombalgia inespecifica sem sinais de alarme, o exame de imagem nas primeiras semanas nao melhora o desfecho e frequentemente revela achados degenerativos assintomaticos que confundem a conduta e aumentam a ansiedade do paciente. A imagem esta indicada quando ha sinal de alarme, deficit progressivo ou consideracao de intervencao. Os sinais de alarme incluem deficit neurologico progressivo, retencao ou incontinencia urinaria ou fecal, anestesia em sela, febre, perda de peso, historia de neoplasia, trauma significativo e dor noturna progressiva. A recomendacao geral e manter atividade dentro do tolerado, ja que repouso prolongado no leito piora a evolucao. A conduta e do medico responsavel."),

"hvn_cauda_equina": (
"Quais achados em lombociatalgia configuram emergencia?",
"O Protocolo NEU-02 identifica a sindrome da cauda equina como emergencia. Os achados que a caracterizam sao retencao ou incontinencia urinaria ou fecal, anestesia em sela e deficit motor bilateral ou progressivo em membros inferiores. Nesse cenario, o atraso na avaliacao tem consequencia funcional permanente, e a conduta e encaminhamento imediato, sem aguardar evolucao ambulatorial. Fora dessa situacao, a lombociatalgia com dor irradiada abaixo do joelho e manobra de elevacao da perna estendida positiva pode ser conduzida com analgesia, manutencao da atividade e reavaliacao, documentando qualquer deficit motor objetivo. A conduta e do medico responsavel."),

# --- Otorrino / Oftalmologia ---------------------------------------------
"hvn_otite_media": (
"Paciente com otalgia. Como conduzir a suspeita de otite media aguda?",
"O Protocolo OTO-01 considera a otoscopia indispensavel: o diagnostico exige abaulamento e hiperemia da membrana timpanica. Dor de ouvido com otoscopia normal sugere outra causa, como disfuncao tubaria ou dor referida — de articulacao temporomandibular, faringe ou dentes. A analgesia e obrigatoria em todos os casos, e a indicacao de antimicrobiano depende da idade, da gravidade e da lateralidade, sendo decisao do medico assistente. Exigem avaliacao imediata os sinais de complicacao: edema, dor e hiperemia retroauricular com deslocamento do pavilhao, que sugerem mastoidite, alem de paralisia facial, vertigem intensa ou sinais meningeos. A prescricao e do medico responsavel."),

"hvn_sinusite_bacteriana": (
"Como diferenciar sinusite viral de bacteriana?",
"O Protocolo OTO-01 lembra que a maioria dos quadros de sinusite aguda e viral. Sugerem origem bacteriana tres padroes: sintomas por mais de dez dias sem melhora; sintomas graves com febre alta e secrecao purulenta por tres a quatro dias consecutivos; ou piora apos melhora inicial, o chamado padrao de dupla piora. Secrecao purulenta isolada, nos primeiros dias, nao diferencia as duas situacoes. Exigem avaliacao imediata os sinais de complicacao: edema periorbitario, alteracao da motricidade ocular ou da acuidade visual, cefaleia intensa, sinais neurologicos ou toxemia. A decisao sobre antimicrobiano segue o Protocolo INF-03 e cabe ao medico responsavel."),

"hvn_conjuntivite_avaliacao": (
"Paciente com olho vermelho e secrecao purulenta. Como avaliar e o que orientar?",
"O Protocolo OFT-01 orienta que a triagem comece pela acuidade visual e pela presenca de dor ocular verdadeira: olho vermelho com visao preservada e sem dor profunda costuma ser conjuntival e benigno. A conjuntivite bacteriana cursa com hiperemia difusa, secrecao purulenta e palpebras aderidas ao despertar, geralmente de inicio unilateral. As orientacoes incluem higiene das maos, toalha individual, compressas, nao compartilhar objetos pessoais, nao ocluir o olho e suspender lente de contato ate a resolucao. Um alerta relevante: colirios com corticoide nao devem ser usados sem avaliacao oftalmologica, pelo risco de agravar ceratite herpetica e elevar a pressao intraocular. A prescricao e do medico responsavel."),

"hvn_olho_vermelho_alarme": (
"Quais sinais em olho vermelho exigem avaliacao oftalmologica de urgencia?",
"O Protocolo OFT-01 lista reducao da acuidade visual, dor ocular intensa, fotofobia acentuada, pupila irregular ou nao reativa, opacificacao da cornea, hiperemia predominantemente perilimbica, trauma ocular, suspeita de corpo estranho e uso de lente de contato acompanhado de dor. Esses achados sugerem comprometimento alem da conjuntiva — ceratite, uveite, glaucoma agudo ou ulcera de cornea — e nao devem ser conduzidos como conjuntivite. O uso de lente de contato com dor merece atencao especial pelo risco de ceratite infecciosa. A avaliacao e a conduta sao do oftalmologista responsavel."),

# --- Dermatologia / Cirurgia ---------------------------------------------
"hvn_dermatite_contato": (
"Paciente com lesao pruriginosa apos uso de produto de limpeza. Como conduzir?",
"O Protocolo DER-01 aponta que a principal pista diagnostica da dermatite de contato e a distribuicao geografica da lesao, acompanhando a area de exposicao ao agente. A anamnese deve investigar produtos de limpeza, cosmeticos, metais como niquel em bijuterias e fivelas, latex, plantas, medicamentos topicos e exposicoes ocupacionais, sempre relacionando o inicio a uma exposicao nova. A conduta e afastar o agente suspeito e adotar cuidados de barreira: hidratacao da pele, luvas na exposicao inevitavel e evitar sabonetes irritantes. Exigem avaliacao imediata acometimento extenso, envolvimento de mucosas, bolhas com descolamento cutaneo, febre e toxemia. O tratamento topico e definido pelo medico responsavel."),

"hvn_reacao_cutanea_grave": (
"Quais achados diferenciam uma reacao cutanea benigna de uma reacao grave a medicamento?",
"O Protocolo DER-01 sinaliza como graves o acometimento cutaneo extenso, o envolvimento de mucosas — oral, ocular ou genital —, bolhas com descolamento da pele, febre e toxemia. Esse conjunto e compativel com reacao cutanea grave a medicamento e exige avaliacao imediata. Separadamente, urticaria disseminada com angioedema, dispneia, estridor, hipotensao ou sintomas gastrointestinais configura anafilaxia, que e emergencia. Em qualquer reacao atribuida a medicamento, o registro explicito no campo de alergias do prontuario, com o nome do agente e a descricao da reacao, e obrigatorio — ver Protocolo SEG-01. A conduta e do medico responsavel."),

"hvn_hernia_inguinal": (
"Paciente com hernia inguinal redutivel e assintomatica. Qual a conduta?",
"O Protocolo CIR-01 classifica esse caso como conduta eletiva. O encaminhamento cirurgico considera sintomas, tamanho, ocupacao do paciente, risco de encarceramento e comorbidades. O exame deve ser feito em pe e deitado, com manobra de esforco. A avaliacao pre-operatoria registra comorbidades, medicamentos em uso — com atencao especial a anticoagulantes e antiagregantes —, alergias e exames pendentes relevantes. A orientacao ao paciente enquanto aguarda a cirurgia e explicita: procurar atendimento imediato se o abaulamento deixar de reduzir, ou se houver dor intensa, vomitos ou parada de eliminacao de gases e fezes. A indicacao cirurgica e do cirurgiao responsavel."),

"hvn_hernia_estrangulada": (
"Quando uma hernia inguinal deixa de ser caso eletivo?",
"O Protocolo CIR-01 define como emergencia cirurgica a hernia irredutivel acompanhada de dor intensa, hiperemia local, vomitos, distensao abdominal e parada de eliminacao de gases e fezes — quadro sugestivo de estrangulamento ou obstrucao intestinal. Nessa situacao, o encaminhamento e imediato e nao se deve insistir em tentativas repetidas de reducao forcada, pelo risco de reduzir alca inviavel para a cavidade. A distincao operacional entre hernia redutivel, encarcerada e estrangulada orienta a urgencia do encaminhamento. Ver tambem o Protocolo URG-01. A conduta cirurgica e do medico responsavel."),

# --- Seguranca e transversais --------------------------------------------
"hvn_sinais_vitais_alerta": (
"Quais combinacoes de sinais vitais devem ser sinalizadas com prioridade a equipe?",
"O Protocolo SEG-02 destaca que combinacoes pesam mais que valores isolados: taquipneia com hipotensao, ou febre com alteracao do nivel de consciencia, sugerem deterioracao clinica e possivel sepse, exigindo avaliacao medica imediata e nao apenas registro. Individualmente, devem ser sinalizados frequencia respiratoria elevada ou muito reduzida, saturacao abaixo de 94 por cento em ar ambiente ou queda em relacao ao basal, hipotensao, taquicardia persistente, bradicardia sintomatica, febre com calafrios, hipotermia e alteracao aguda do nivel de consciencia. A comparacao com os valores anteriores do proprio paciente e mais informativa que o valor absoluto. A avaliacao e a conduta sao do medico responsavel."),

"hvn_exames_pendentes": (
"Por que exames pendentes sao destacados antes de qualquer sugestao de conduta?",
"O Protocolo SEG-03 trata o exame solicitado e nao avaliado como uma das principais fontes de falha assistencial evitavel. Por isso a equipe deve ser alertada sobre tipo, data de solicitacao e medico solicitante antes de qualquer decisao. Sugerir conduta definitiva ignorando um exame diagnostico em andamento e um erro de raciocinio clinico. Tem prioridade critica biopsia e anatomopatologico, imagem para investigacao de suspeita de neoplasia, culturas em paciente em uso de antimicrobiano, exames em suspeita de trombose ou embolia, e reavaliacao de imagem em paciente oncologico. Resultado com achado critico exige comunicacao ativa ao solicitante, nao apenas registro. A conduta e do medico responsavel."),

"hvn_polifarmacia": (
"Paciente em uso de multiplos medicamentos continuos. Que cuidados o hospital orienta?",
"O Protocolo SEG-01 considera que pacientes com cinco ou mais medicamentos de uso continuo tem risco elevado de interacao e de cascata terapeutica, em que um efeito adverso e tratado como novo diagnostico. A recomendacao e revisao periodica da lista completa, incluindo fitoterapicos e suplementos, que costumam ser omitidos pelo paciente quando a pergunta e feita apenas sobre remedios. Merecem atencao prioritaria as associacoes de anticoagulantes com anti-inflamatorios e com antimicrobianos, antiagregantes em paciente com procedimento programado, medicamentos que prolongam o intervalo QT e o uso concomitante de multiplos depressores do sistema nervoso central. A revisao terapeutica e do medico responsavel."),

"hvn_checkup_rotina": (
"Paciente assintomatico em consulta de rotina pede um painel amplo de exames. Como orientar?",
"O Protocolo ADM-01 parte de uma distincao: rastrear e diferente de investigar. Exames indiscriminados em paciente assintomatico produzem achados incidentais que geram ansiedade, investigacao adicional e risco, sem beneficio demonstrado. A consulta de rotina deve cobrir historia familiar, habitos, medicamentos e suplementos, saude mental, situacao vacinal, pressao arterial, peso e altura, e considerar os rastreamentos consolidados conforme idade, sexo e fatores de risco. As intervencoes de maior impacto sao cessacao do tabagismo, atividade fisica regular, atualizacao vacinal e controle de pressao e glicemia — mais relevantes que ampliar o painel laboratorial. A indicacao de exames e do medico responsavel."),

"hvn_papel_do_assistente": (
"Qual o papel do assistente virtual no atendimento e quais os seus limites?",
"O assistente e ferramenta de apoio a decisao clinica: organiza informacao do prontuario, destaca pontos relevantes e de atencao, sinaliza exames pendentes e recupera protocolos internos, sempre indicando a fonte consultada. Ele nao prescreve tratamento, medicamento ou dose, e nao substitui a avaliacao do medico. Toda sugestao e enquadrada como recomendacao a ser validada por um profissional responsavel, e respostas que contenham linguagem de prescricao direta sao sinalizadas para revisao humana antes de qualquer uso. Quando o contexto disponivel nao for suficiente para responder, o assistente deve declarar isso explicitamente, em vez de completar a informacao que falta."),
}


def escrever_protocolos() -> int:
    PROTOCOLOS.mkdir(parents=True, exist_ok=True)
    for stem, corpo in PROTOCOLOS_NOVOS.items():
        (PROTOCOLOS / f"{stem}.md").write_text(
            f"# {stem}\n\n{corpo.strip()}\n", encoding="utf-8"
        )
    return len(PROTOCOLOS_NOVOS)


def escrever_faqs() -> int:
    FAQS.mkdir(parents=True, exist_ok=True)
    for stem, (instrucao, resposta) in FAQS_NOVAS.items():
        (FAQS / f"{stem}.md").write_text(
            f"## Instrução\n{instrucao.strip()}\n\n## Resposta\n{resposta.strip()}\n",
            encoding="utf-8",
        )
    return len(FAQS_NOVAS)


if __name__ == "__main__":
    n_prot = escrever_protocolos()
    n_faq = escrever_faqs()
    print(f"Protocolos escritos: {n_prot} -> {PROTOCOLOS}")
    print(f"FAQs escritas: {n_faq} -> {FAQS}")
