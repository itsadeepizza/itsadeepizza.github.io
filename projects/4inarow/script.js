var gameField = new Array();
var board = document.getElementById("game-table");
var message = document.getElementById("message");
var currentCol;
var currentRow;
var currentPlayer;
var id = 1;
var loading = 0

class MobileNet {
    constructor() { }

    async loadMobilenet() {
        this.model = await tflite.loadTFLiteModel('./little_group.tflite');
        console.log("Model loaded")
        return true
    }

    predict() {
        let state = get_state()
        let out = mobileNet.model.predict(state);
        return out.argMax([-1]).dataSync()
    }
}

var nn_move

mobileNet = new MobileNet()
mobileNet.loadMobilenet().then(result => newgame())


function get_state(){
    //Get the board state in javascript, make some processing for campatibility and convert to tensor
    let state_list = [];
    for(let i=0; i<6; i++){
        state_list[i] = new Array();
        for(let j=0; j<7; j++){
            //vertically flip as in python library
            let token = gameField[5-i][j];
            // Change token values to 1 (AI player) or -1 (human player, opponent)
            var new_token = 0
            if (token == 1) {
                new_token = -1
            }
            if (token == 2) {
                new_token = 1
            }
            state_list[i].push(new_token);
        }
    }
    console.log(state_list)
    let state = tf.expandDims(tf.tensor(state_list), 0);
    return state
}


function newgame(){
  board.innerHTML = "";
  id = 1;
  prepareField();
  placeDisc(Math.floor(Math.random()*2)+1)
}

function checkForVictory(row,col){
  if(getAdj(row,col,0,1)+getAdj(row,col,0,-1) > 2) {
      return true;
  }
  if(getAdj(row,col,1,0) > 2) {
      return true;
  }
  if(getAdj(row,col,-1,1)+getAdj(row,col,1,-1) > 2) {
      return true;
  }
  if(getAdj(row,col,1,1)+getAdj(row,col,-1,-1) > 2) {
      return true;
  }
  return false;

}

function getAdj(row,col,row_inc,col_inc){
  // A recursive function which count consecutive token of the same player in a given direction
  if((cellVal(row,col) > 0) && (cellVal(row,col) == cellVal(row+row_inc,col+col_inc))){
    return 1+getAdj(row+row_inc,col+col_inc,row_inc,col_inc);
  } else {
    return 0;
  }
}

function cellVal(row,col){
  //  return gameField if well defined, otherwise -1 (if calling outside the board)
  if(gameField[row] == undefined || gameField[row][col] == undefined){
    return -1;
  } else {
    return gameField[row][col];
  }
}

function firstFreeRow(col,player){
  for(var i = 0; i<6; i++){
    if(gameField[i][col]!=0){
      break;
    }
  }
  gameField[i-1][col] = player;
  return i-1;
}

function possibleColumns(){
  var moves_array = new Array();
  for(var i=0; i<7; i++){
    if(gameField[0][i] == 0){
      moves_array.push(i);
    }
  }
  return moves_array;
}

function think(){
  // the AI choose the move using the model
  message.innerHTML = "AI is thinking ..."
  nn_move = mobileNet.predict()[0]
  console.log(nn_move)
  return nn_move;
}

function Disc(player){
  this.player = player;
  this.color = player == 1 ? 'red' : 'yellow';
  this.id = id.toString();
  id++;
  
  this.addToScene = function(){
    board.innerHTML += '<div id="d'+this.id+'" class="disc '+this.color+'"></div>';
    if(currentPlayer==2){
      //computer move
      currentCol = think();
      document.getElementById('d'+this.id).style.left = (8+62*currentCol)+"px";
      dropDisc(this.id,currentPlayer);
    }
    else{
        currentCol = 0
        loading = 1
        message.innerHTML = "It's your turn! - Click here to reset"
    }
  }
  
  var $this = this;
  document.onmousemove = function(evt){
    if(currentPlayer == 1){
    currentCol = Math.floor((evt.clientX - board.offsetLeft)/62);
    if(currentCol<0){currentCol=0;}
    if(currentCol>6){currentCol=6;}
    if (loading == 1){
        loading = 2
    }
    document.getElementById('d'+$this.id).style.left = (7+62*currentCol)+"px";
    document.getElementById('d'+$this.id).style.top = "-55px";
    }
  }
  document.onload = function(evt){
    if(currentPlayer == 1){
    currentCol = Math.floor((evt.clientX - board.offsetLeft)/62);
    if(currentCol<0){currentCol=0;}
    if(currentCol>6){currentCol=6;}
    document.getElementById('d'+$this.id).style.left = (7+62*currentCol)+"px";
    document.getElementById('d'+$this.id).style.top = "-55px";
    }
  }
  
  document.onclick = function(evt){
    if(currentPlayer == 1 && loading == 2){

      if(possibleColumns().indexOf(currentCol) != -1){
        dropDisc($this.id,$this.player);
      }
    }
  }
}

function dropDisc(cid,player){
  currentRow = firstFreeRow(currentCol,player);
  console.log("current", player, currentRow, currentCol)
  moveit(cid,(9+currentRow*62));

  console.log(id)

  if (! checkForMoveVictory()){
    currentPlayer = 3 - player;
    var disc = new Disc(currentPlayer);
    disc.addToScene()
    }
  else{
      $this = ""
  }


}

message.onmouseup=function (){
    loading = true

    newgame()
}


function checkForMoveVictory(){
  //  Check if a player has win with last move

  if(checkForVictory(currentRow,currentCol)) {
    var ww = currentPlayer == 2 ? 'AI' : 'Player';
    message.innerHTML = ww+" win! Click here to play again"
    return true
    // alert(ww+" win!");
    // board.innerHTML = "";
    // newgame();
  }
  if (id == 44){
      message.innerHTML = "Draw! Click here to play again"
      return true
  }
  return false
}

function placeDisc(player){
  currentPlayer = player;
  var disc = new Disc(player);
  disc.addToScene();
}

function prepareField(){
  gameField = new Array();
  for(var i=0; i<6; i++){
    gameField[i] = new Array();
    for(var j=0; j<7; j++){
      gameField[i].push(0);
    }
  }
}

function moveit(who,where){
    document.getElementById('d'+who).style.top = where+'px';
}

function delay(time) {
  return new Promise(resolve => setTimeout(resolve, time));
}