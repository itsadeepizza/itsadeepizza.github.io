var gameField = new Array();
var board = document.getElementById("game-table");
var currentCol;
var currentRow;
var currentPlayer;
var id = 1;

class MobileNet {
    constructor() { }

    async loadMobilenet() {
        this.model = await tflite.loadTFLiteModel('/production/little_group.tflite');
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

newgame();

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
  prepareField();
  mobileNet = new MobileNet()
  mobileNet.loadMobilenet().then(result => placeDisc(Math.floor(Math.random()*2)+1))

}

function checkForVictory(row,col){
  if(getAdj(row,col,0,1)+getAdj(row,col,0,-1) > 2){
    return true;
  } else {
    if(getAdj(row,col,1,0) > 2){
      return true;
    } else {
      if(getAdj(row,col,-1,1)+getAdj(row,col,1,-1) > 2){
        return true;
      } else {
        if(getAdj(row,col,1,1)+getAdj(row,col,-1,-1) > 2){
          return true;
        } else {
          return false;
        }
      }
    }
  }
}

function getAdj(row,col,row_inc,col_inc){
  if(cellVal(row,col) == cellVal(row+row_inc,col+col_inc)){
    return 1+getAdj(row+row_inc,col+col_inc,row_inc,col_inc);
  } else {
    return 0;
  }
}

function cellVal(row,col){
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

  // Convert to list for compatibility
  nn_move = [mobileNet.predict()[0]]
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
      var possibleMoves = think();
      var cpuMove = Math.floor( Math.random() * possibleMoves.length);
      currentCol = possibleMoves[cpuMove];
      document.getElementById('d'+this.id).style.left = (14+60*currentCol)+"px";
      dropDisc(this.id,currentPlayer);
    }
  }
  
  var $this = this;
  document.onmousemove = function(evt){
    if(currentPlayer == 1){
    currentCol = Math.floor((evt.clientX - board.offsetLeft)/60);
    if(currentCol<0){currentCol=0;}
    if(currentCol>6){currentCol=6;}
    document.getElementById('d'+$this.id).style.left = (14+60*currentCol)+"px";
    document.getElementById('d'+$this.id).style.top = "-55px";
    }
  }
  document.onload = function(evt){
    if(currentPlayer == 1){
    currentCol = Math.floor((evt.clientX - board.offsetLeft)/60);
    if(currentCol<0){currentCol=0;}
    if(currentCol>6){currentCol=6;}
    document.getElementById('d'+$this.id).style.left = (14+60*currentCol)+"px";
    document.getElementById('d'+$this.id).style.top = "-55px";
    }
  }
  
  document.onclick = function(evt){
    if(currentPlayer == 1){
      if(possibleColumns().indexOf(currentCol) != -1){
        dropDisc($this.id,$this.player);
      }
    }
  }
}

function dropDisc(cid,player){
  currentRow = firstFreeRow(currentCol,player);
  moveit(cid,(14+currentRow*60));
  currentPlayer = player;
  checkForMoveVictory();
}

function checkForMoveVictory(){
  if(!checkForVictory(currentRow,currentCol)){
    placeDisc(3-currentPlayer);
  } else {
    var ww = currentPlayer == 2 ? 'Computer' : 'Player';
    placeDisc(3-currentPlayer);
    alert(ww+" win!");
    board.innerHTML = "";
    newgame();
  }
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